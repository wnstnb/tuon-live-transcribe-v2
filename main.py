import asyncio
import websockets
import websockets.server
import logging
import os
import json
import threading
from typing import Type, AsyncGenerator

import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    # StreamingSessionParameters, # Less likely to be used in this server model
    TerminationEvent,
    TurnEvent,
)
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ASSEMBLYAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Queue to pass audio chunks from async WebSocket handler to sync AssemblyAI thread
# One queue per connection, so defined inside client_handler_wrapper

# Sync iterator that pulls from an asyncio.Queue
# This runs in the AssemblyAI thread and blocks until data is available in the queue
def audio_chunk_sync_iterator(audio_queue: asyncio.Queue, main_loop: asyncio.AbstractEventLoop):
    logger.info("audio_chunk_sync_iterator started")
    while True:
        future = asyncio.run_coroutine_threadsafe(audio_queue.get(), main_loop)
        chunk = future.result()  # Blocks until item is available

        if chunk is None:  # Sentinel value to indicate end of stream
            logger.info("audio_chunk_sync_iterator received None, ending.")
            main_loop.call_soon_threadsafe(audio_queue.task_done)
            break
        
        logger.debug(f"audio_chunk_sync_iterator got chunk of size {len(chunk)}")
        yield chunk
        main_loop.call_soon_threadsafe(audio_queue.task_done)
    logger.info("audio_chunk_sync_iterator finished")


def assemblyai_processor_thread(
    websocket_connection: websockets.server.ServerProtocol,
    audio_queue: asyncio.Queue, 
    main_event_loop: asyncio.AbstractEventLoop
):
    logger.info(f"AssemblyAI processor thread started for client {websocket_connection.remote_address}")
    
    aai_client = StreamingClient(
        StreamingClientOptions(
            api_key=api_key,
            # api_host can be specified if needed, defaults usually work
        )
    )

    def send_message_to_client_threadsafe(message_data: dict):
        async def _send():
            try:
                await websocket_connection.send(json.dumps(message_data))
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"Client {websocket_connection.remote_address} connection closed. Cannot send message.")
            except Exception as e:
                logger.error(f"Error sending message to client {websocket_connection.remote_address}: {e}")
        
        asyncio.run_coroutine_threadsafe(_send(), main_event_loop)

    # Define AssemblyAI event handlers (these run in the AssemblyAI thread)
    def on_begin(self_client: StreamingClient, event: BeginEvent):
        logger.info(f"AAI Session started for {websocket_connection.remote_address}: {event.id}")
        send_message_to_client_threadsafe({"type": "session_begin", "session_id": event.id})

    def on_turn(self_client: StreamingClient, event: TurnEvent):
        transcript_text = event.transcript
        
        # Log every turn event received from AssemblyAI for debugging
        logger.info(
            f"AAI TurnEvent for {websocket_connection.remote_address}: "
            f"Text='{transcript_text}', "
            f"EndOfTurn={event.end_of_turn}, "
            f"TurnIsFormatted={event.turn_is_formatted}"
        )

        message_to_client = {
            "type": "transcript_update",
            "text": transcript_text,
            # Other relevant event attributes like 'confidence' could be added here if needed by the client
        }

        if event.end_of_turn:
            if event.turn_is_formatted:
                # This is a formatted, final transcript for the utterance.
                message_to_client["is_final"] = True
                logger.info(f"Sending FORMATTED FINAL transcript to client: '{transcript_text}'")
                send_message_to_client_threadsafe(message_to_client)
            else:
                # This is an unformatted, final transcript.
                # We will NOT send this to the client with is_final: true, 
                # as we expect a formatted version to follow because format_turns=True.
                logger.info(f"Received UNFORMATTED FINAL transcript. Waiting for formatted version: '{transcript_text}'")
                # Optionally, send as an interim update if the client should display it while waiting for formatting:
                # message_to_client["is_final"] = False 
                # send_message_to_client_threadsafe(message_to_client)
        else:
            # This is an interim transcript
            message_to_client["is_final"] = False
            send_message_to_client_threadsafe(message_to_client)

    def on_terminated(self_client: StreamingClient, event: TerminationEvent):
        logger.info(f"AAI Session terminated for {websocket_connection.remote_address}: {event.audio_duration_seconds}s audio processed")
        send_message_to_client_threadsafe({
            "type": "session_terminated",
            "audio_duration_seconds": event.audio_duration_seconds
        })

    def on_error(self_client: StreamingClient, error: StreamingError):
        logger.error(f"AAI Error for {websocket_connection.remote_address}: {error}")
        send_message_to_client_threadsafe({"type": "error", "message": str(error)})

    aai_client.on(StreamingEvents.Begin, on_begin)
    aai_client.on(StreamingEvents.Turn, on_turn)
    aai_client.on(StreamingEvents.Termination, on_terminated)
    aai_client.on(StreamingEvents.Error, on_error)

    try:
        logger.info(f"Connecting AssemblyAI client for {websocket_connection.remote_address}...")
        aai_client.connect(
            StreamingParameters(
                sample_rate=16000,  # Ensure client sends audio at this rate
                format_turns=True,
                end_utterance_silence_threshold=1500  # Example: 1200ms (1.2 seconds) of silence
            )
        )
        logger.info(f"AssemblyAI client connected for {websocket_connection.remote_address}. Starting stream processing.")
        
        # Create the synchronous iterator for the audio chunks
        audio_iterator = audio_chunk_sync_iterator(audio_queue, main_event_loop)
        aai_client.stream(audio_iterator)
        
        logger.info(f"AssemblyAI stream processing ended for {websocket_connection.remote_address}.")

    except Exception as e:
        logger.error(f"Exception in AssemblyAI processor thread for {websocket_connection.remote_address}: {e}", exc_info=True)
        send_message_to_client_threadsafe({"type": "error", "message": f"Server-side AAI error: {e}"})
    finally:
        logger.info(f"Disconnecting AssemblyAI client for {websocket_connection.remote_address}.")
        aai_client.disconnect(terminate=True)
        # Ensure the queue is unblocked if an error occurred before stream iterator finished
        asyncio.run_coroutine_threadsafe(audio_queue.put(None), main_event_loop)


async def client_connection_handler(websocket: websockets.server.ServerProtocol, path: str = None):
    client_address = websocket.remote_address
    logger.info(f"Client connected: {client_address}, path: {path}")
    
    audio_data_queue = asyncio.Queue()
    main_loop = asyncio.get_running_loop()

    # Start the AssemblyAI processing in a separate thread
    aai_thread = threading.Thread(
        target=assemblyai_processor_thread,
        args=(websocket, audio_data_queue, main_loop),
        daemon=True # Ensure thread exits when main program exits
    )
    aai_thread.start()

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                logger.debug(f"Received audio chunk from {client_address}, size: {len(message)}")
                await audio_data_queue.put(message)
            elif isinstance(message, str):
                logger.info(f"Received text message from {client_address}: {message}")
                if message.upper() == "END_STREAM":
                    logger.info(f"END_STREAM signal received from {client_address}. Signaling AAI thread.")
                    await audio_data_queue.put(None) # Signal end of audio
                    # break # Optionally break here, or wait for AAI to terminate
                # Handle other text messages if needed
            else:
                logger.warning(f"Received unexpected message type from {client_address}: {type(message)}")
        
        logger.info(f"Client {client_address} stopped sending messages or WebSocket closed by client.")
        # Ensure AAI thread is signaled to stop if not already
        await audio_data_queue.put(None)

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client {client_address} disconnected gracefully (OK).")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client {client_address} connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler for {client_address}: {e}", exc_info=True)
    finally:
        logger.info(f"Cleaning up for client {client_address}.")
        # Signal end of audio stream to the AssemblyAI thread if not already done
        await audio_data_queue.put(None)
        
        if aai_thread.is_alive():
            logger.info(f"Waiting for AssemblyAI thread to join for client {client_address}...")
            # audio_data_queue.join() # Wait for all items to be processed
            # Instead of queue.join(), rely on thread.join() with a timeout
            aai_thread.join(timeout=5.0) # Wait for the thread to finish
            if aai_thread.is_alive():
                 logger.warning(f"AssemblyAI thread for {client_address} did not join in time.")
        logger.info(f"Client handler for {client_address} finished.")


async def start_server():
    if not api_key:
        logger.critical("ASSEMBLYAI_API_KEY is not set. Cannot start server.")
        return

    host = "0.0.0.0"  # Listen on all available interfaces
    port = int(os.getenv("PORT", 8000))  # Use Render's PORT or default to 8000
    
    logger.info(f"Starting WebSocket server on ws://{host}:{port}")
    async with websockets.serve(client_connection_handler, host, port):
        await asyncio.Future()  # Run forever until interrupted

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("Server shutting down due to KeyboardInterrupt...")
    except Exception as e:
        logger.critical(f"Failed to start server: {e}", exc_info=True)
