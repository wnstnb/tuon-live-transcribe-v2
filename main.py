import asyncio
import websockets
import websockets.server
import websockets.http11 # For Request, read_headers, parse_path
from websockets.datastructures import Headers as HTTPHeadersDef # Alias for type hints
from websockets.exceptions import InvalidMessage, NegotiationError
from websockets.typing import Origin, Subprotocol # For type hints
import logging
import os
import json
import threading
from typing import Type, AsyncGenerator, Tuple, Dict, Optional, List, Sequence, Callable

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

# Custom Server Connection Protocol to handle HEAD requests for health checks
class HealthCheckFriendlyServerConnection(websockets.asyncio.server.ServerConnection):
    async def read_http_request(self) -> Tuple[str, websockets.datastructures.Headers]:
        """
        Override to allow HEAD /healthz to pass initial parsing,
        so process_request can handle it.
        """
        try:
            request_line_bytes = await self.reader.read_line(self.max_line_length)
            request_line = request_line_bytes.decode()
        except EOFError as exc:
            raise InvalidMessage("connection closed before receiving a request") from exc
        except UnicodeDecodeError as exc:
            # Log the problematic bytes if possible and safe
            logger.warning(f"Invalid UTF-8 in HTTP request line. Bytes: {request_line_bytes!r}", exc_info=False)
            raise InvalidMessage(f"invalid UTF-8 in HTTP request line: {exc}") from exc

        if not request_line.endswith("\\r\\n"):
            # Log the actual line received for debugging
            logger.warning(f"HTTP request line not CRLF-terminated: {request_line!r}")
            raise InvalidMessage("HTTP request line isn't terminated by CRLF")
        
        processed_request_line = request_line[:-2]

        try:
            method, raw_path, version = processed_request_line.split(" ", 2)
        except ValueError:
            logger.warning(f"Invalid HTTP request line format: {processed_request_line!r}")
            raise InvalidMessage(f"invalid HTTP request line: {processed_request_line!r}")

        # Basic version check (websockets primarily targets HTTP/1.1 for WebSocket handshake)
        if version != "HTTP/1.1": # Stricter than original for simplicity here, original allows HTTP/2.0
             # Log the version for diagnostics
            logger.warning(f"Unsupported HTTP version: {version} from request: {processed_request_line!r}")
            if version.startswith("HTTP/"):
                 raise NegotiationError(f"unsupported HTTP version: {version}")
            else:
                 raise InvalidMessage(f"invalid HTTP version: {version}")

        _parsed_headers = await websockets.http11.read_headers(
            self.reader.read_line, 
            max_header_count=self.max_header_count, 
            max_line_length=self.max_line_length
        )
        _parsed_path = websockets.http11.parse_path(raw_path)

        # Allow HEAD /healthz to pass this stage.
        # process_request will then be called by the handshake() method.
        if method == "HEAD" and _parsed_path == "/healthz":
            self.request_headers = _parsed_headers # Used by handshake() to pass to process_request
            logger.info(f"Allowing HEAD /healthz through read_http_request. Path: {_parsed_path}, Headers: {_parsed_headers}")
            return _parsed_path, _parsed_headers

        # For all other requests, enforce GET method for WebSocket handshake.
        if method != "GET":
            logger.warning(f"Unsupported HTTP method for WebSocket: {method}. Path: {_parsed_path}, Request: {processed_request_line!r}")
            raise InvalidMessage(f"unsupported HTTP method; expected GET (or HEAD for /healthz); got {method}")

        # It's a GET request, proceed as normal for WebSocket or process_request handling.
        self.request_headers = _parsed_headers
        logger.debug(f"Allowing GET request through read_http_request. Path: {_parsed_path}, Headers: {_parsed_headers}")
        return _parsed_path, _parsed_headers


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


async def process_http_request(path: str, request_headers: websockets.datastructures.Headers) -> Optional[Tuple[int, List[Tuple[str, str]], bytes]]:
    logger.info(f"HTTP Request Received: Path='{path}', Method (from headers if present)='{request_headers.get('Method', 'N/A')}', All Headers='{request_headers}'")

    if path == "/healthz":
        logger.info(f"Health check path '/healthz' matched. Responding 200 OK.")
        # For HEAD requests, the body is ignored by the client, but sending it is fine.
        # For GET requests, this body will be sent.
        return (200, [("Content-Type", "text/plain"), ("Connection", "close")], b"OK")
    
    logger.debug(f"Path '{path}' not /healthz, proceeding to WebSocket handshake attempt.")
    return None # Proceed with WebSocket handling for other paths

async def start_server():
    if not api_key:
        logger.critical("ASSEMBLYAI_API_KEY is not set. Cannot start server.")
        return

    host = "0.0.0.0"  # Listen on all available interfaces
    port = int(os.getenv("PORT", 8000))  # Use Render's PORT or default to 8000
    
    logger.info(f"Starting WebSocket server on ws://{host}:{port} with /healthz endpoint and custom connection handler")
    async with websockets.serve(
        client_connection_handler, 
        host, 
        port,
        process_request=process_http_request,
        create_protocol=HealthCheckFriendlyServerConnection  # Use our custom protocol
    ):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("Server shutting down due to KeyboardInterrupt...")
    except Exception as e:
        logger.critical(f"Failed to start server: {e}", exc_info=True)
