import asyncio
import websockets
import json
import base64
import logging
import pyaudio
import signal # For graceful shutdown

# Configure logging for the client
logging.basicConfig(level=logging.INFO, format='%(asctime)s CLIENT: %(levelname)s %(message)s')

GATEWAY_URL = "ws://localhost:8765"

# PyAudio Configuration - must match AssemblyAI expectations
FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1              # Mono
RATE = 16000              # 16kHz sample rate
CHUNK_DURATION_MS = 50    # Duration of each audio chunk in milliseconds
FRAMES_PER_BUFFER = int(RATE * (CHUNK_DURATION_MS / 1000.0)) # Number of frames per buffer

# Global flag to signal shutdown
shutdown_flag = asyncio.Event()

def signal_handler(sig, frame):
    logging.info("Shutdown signal received. Cleaning up...")
    shutdown_flag.set()

async def record_and_send(websocket):
    """Records audio from microphone and sends it to the WebSocket gateway."""
    audio_interface = pyaudio.PyAudio()
    try:
        stream = audio_interface.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER
        )
        logging.info("Microphone stream opened. Recording...")
        logging.info("Speak into your microphone. Press Ctrl+C to stop.")

        while not shutdown_flag.is_set():
            try:
                # Read audio data from the stream
                audio_data_bytes = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                
                # Send raw audio bytes directly
                await websocket.send(audio_data_bytes)
                # logging.debug(f"Sent audio chunk of {len(audio_data_bytes)} bytes")
                await asyncio.sleep(0.01) # Small sleep to yield control, adjust if needed
            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    logging.warning("Input overflowed. Frame dropped.")
                else:
                    logging.error(f"PyAudio IOError: {e}")
                    break # Exit loop on other IOErrors
            except Exception as e:
                logging.error(f"Error in recording/sending loop: {e}")
                break

    except Exception as e:
        logging.error(f"Could not open microphone stream: {e}")
        return # Exit if microphone can't be opened
    finally:
        logging.info("Stopping microphone stream...")
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        audio_interface.terminate()
        logging.info("Microphone stream closed and PyAudio terminated.")
        
        # Signal end of audio to gateway
        if websocket.open:
            try:
                end_stream_signal = "END_STREAM"
                await websocket.send(end_stream_signal)
                logging.info(f"Sent '{end_stream_signal}' signal to gateway.")
            except Exception as e:
                logging.error(f"Error sending end_stream signal: {e}")

async def receive_messages(websocket):
    """Receives and prints messages from the WebSocket gateway."""
    try:
        async for message_str in websocket:
            try:
                message_json = json.loads(message_str)
                msg_type = message_json.get("type")

                if msg_type == "transcript_update":
                    text = message_json.get('text', '')
                    is_final = message_json.get('is_final', False)
                    final_indicator = "[FINAL]" if is_final else "[PARTIAL]"
                    logging.info(f"TRANSCRIPT {final_indicator}: {text}")
                elif msg_type == "session_begin":
                    session_id = message_json.get('session_id', 'N/A')
                    logging.info(f"SESSION BEGAN: ID {session_id}")
                elif msg_type == "session_terminated":
                    duration = message_json.get('audio_duration_seconds', 'N/A')
                    logging.info(f"SESSION TERMINATED: Processed {duration}s of audio.")
                elif msg_type == "error":
                    error_message = message_json.get('message', 'Unknown error')
                    logging.error(f"SERVER ERROR: {error_message}")
                else:
                    logging.info(f"GATEWAY: {message_json}")
            except json.JSONDecodeError:
                logging.warning(f"Received non-JSON message from gateway: {message_str}")
            except Exception as e:
                logging.error(f"Error processing received message: {e}")
    except websockets.exceptions.ConnectionClosed as e:
        logging.info(f"Connection to gateway closed: {e}")
    except Exception as e:
        logging.error(f"Error in receiving loop: {e}")
    finally:
        logging.info("Receive loop finished.")

async def run_test_client_microphone():
    """Connects to the gateway, streams microphone audio, and prints responses."""
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        async with websockets.connect(GATEWAY_URL) as websocket:
            logging.info(f"Connected to gateway: {GATEWAY_URL}")

            # Create tasks for recording/sending and receiving
            send_task = asyncio.create_task(record_and_send(websocket))
            receive_task = asyncio.create_task(receive_messages(websocket))

            # Wait for either task to complete or shutdown signal
            done, pending = await asyncio.wait(
                [send_task, receive_task, asyncio.create_task(shutdown_flag.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )

            # If shutdown_flag caused completion, ensure other tasks are cancelled
            if shutdown_flag.is_set():
                logging.info("Shutdown initiated, cancelling tasks...")
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            
            # Await all tasks to ensure cleanup (especially send_task for end_stream)
            await send_task
            await receive_task
            
            logging.info("All tasks completed.")

    except ConnectionRefusedError:
        logging.error(f"Connection refused. Is the gateway (main.py) running at {GATEWAY_URL}?")
    except websockets.exceptions.InvalidURI:
        logging.error(f"Invalid WebSocket URI: {GATEWAY_URL}")
    except Exception as e:
        logging.error(f"An error occurred in run_test_client_microphone: {e}", exc_info=True)
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler) # Restore original SIGINT handler
        logging.info("Test client finished.")

if __name__ == "__main__":
    asyncio.run(run_test_client_microphone()) 