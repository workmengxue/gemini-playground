"""
Gemini Multimodal Demo with Requests

Purpose:
  This script demonstrates multimodal interaction with Google's Gemini API
  using the `requests` library for HTTP communication. It can send text,
  a video frame captured from an RTSP stream, and raw PCM audio data (via a socket server)
  to the Gemini model and process the response (text and potentially output audio).

Dependencies:
  - requests: For making HTTP requests to the Gemini API.
  - opencv-python: For capturing video frames from an RTSP stream.
  Install them using pip:
    pip install requests opencv-python

Configuration:

1. GEMINI_API_KEY:
   - REQUIRED. Your Google Gemini API key.
   - Set it as an environment variable named `GEMINI_API_KEY`.
   - If not found in the environment, the script will prompt you to enter it.

2. MODEL_NAME:
   - The Gemini model to use (e.g., "gemini-1.5-flash-latest", "gemini-1.5-pro-latest").
   - Default: "gemini-1.5-flash-latest"

3. RTSP_URL:
   - The URL of the RTSP video stream you want to capture a frame from.
   - Example: 'rtsp://username:password@your_camera_ip/stream_path'
   - Default: 'rtsp://192.168.66.130/main_ch' (replace with your actual RTSP URL)

4. Audio Input Server:
   - The script starts a TCP socket server to listen for incoming raw PCM audio.
   - AUDIO_SERVER_HOST: Host for the audio server. '0.0.0.0' listens on all interfaces.
     Default: '0.0.0.0'
   - AUDIO_SERVER_PORT: Port for the audio server.
     Default: 6790
   - AUDIO_SAMPLE_RATE: Sample rate of the incoming PCM audio. MUST match the source.
     Default: 8000 (Hz)
   - AUDIO_CHANNELS: Number of audio channels. MUST match the source.
     Default: 1 (mono)
   - AUDIO_DTYPE: Data type of the incoming PCM audio samples.
     Default: 'int16' (signed 16-bit integers)
     This configuration corresponds to 'audio/L16' MIME type for Gemini.

Sending Audio to the Script:
  You need to send raw PCM audio data that matches the configured sample rate,
  channels, and data type (signed 16-bit integer Little Endian is typical for .raw or .pcm files).

  Example using `netcat` (nc) on Linux/macOS:
  1. Ensure you have a raw audio file (e.g., `audio.raw`) in the correct format.
     You can create one with `ffmpeg`:
     `ffmpeg -i your_audio_file.mp3 -f s16le -ar 8000 -ac 1 audio.raw`
     (Adjust -ar and -ac to match AUDIO_SAMPLE_RATE and AUDIO_CHANNELS if changed)
  2. Once the script's audio server is running (it will print a message),
     run the following command in a separate terminal:
     `nc <AUDIO_SERVER_HOST> <AUDIO_SERVER_PORT> < audio.raw`
     (Replace <AUDIO_SERVER_HOST> and <AUDIO_SERVER_PORT> with the script's actual values,
      e.g., `nc localhost 6790 < audio.raw` if running on the same machine).
     The audio data will be sent once, and `nc` will close the connection. The script
     will then use this audio for the *next* Gemini request you trigger by entering text.

Running the Script:
  1. Configure the variables above, especially `GEMINI_API_KEY` (as env var or be ready to paste it)
     and `RTSP_URL`.
  2. Run the script: `python gemini_multimodal_demo.py`
  3. The script will:
     - Attempt to start the audio input server.
     - Prompt you to enter a text query for Gemini.
     - Attempt to capture a frame from the `RTSP_URL`.
     - If you have sent audio via the socket, it will use the last received audio.
     - Send the combined inputs to Gemini.
     - Print Gemini's text response and save any audio output from Gemini to a file.
  4. To send new audio for a subsequent turn (if you modify the script for multiple turns),
     you'll need to run the `nc` command again.

Troubleshooting:
  - "Error: Could not open RTSP stream": Check your RTSP_URL, camera credentials, and network.
  - Audio server errors: Ensure the port is not in use. Verify the client is sending compatible PCM data.
  - Gemini API errors: Check your API key and ensure the model name is correct.
"""
import os
import requests
import json
import base64
import cv2  # OpenCV for video
import socket
import threading
import time # For potential delays or timeouts

# --- Configuration ---
# 1. Gemini API Key (REQUIRED)
# Set as environment variable GEMINI_API_KEY or enter when prompted.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 2. Gemini Model Name
# Recommend using a model that supports multimodal input.
MODEL_NAME = "gemini-1.5-flash-latest" # Or "gemini-1.5-pro-latest"

# 3. Video Configuration
RTSP_URL = 'rtsp://192.168.66.130/main_ch' # REPLACE with your actual RTSP stream URL

# 4. Audio Input Configuration (Socket Server)
AUDIO_SERVER_HOST = '0.0.0.0'  # Listen on all available interfaces
AUDIO_SERVER_PORT = 6790       # Port for receiving PCM audio
AUDIO_SAMPLE_RATE = 8000       # Sample rate of the incoming PCM audio (e.g., 8000, 16000, 44100)
AUDIO_CHANNELS = 1             # Number of audio channels (1 for mono, 2 for stereo)
AUDIO_DTYPE = 'int16'          # Data type of incoming PCM (signed 16-bit integer)
                               # This translates to 'audio/L16' (Linear 16-bit PCM) for Gemini

# --- Global Variables ---
# This global variable will store the latest received audio data (base64 encoded)
latest_base64_audio_data = None
audio_server_running = True


# --- Function Definitions ---

def get_gemini_api_key():
    """
    Gets the Gemini API key.
    1. Tries the global GEMINI_API_KEY (populated from environment variable).
    2. If not found, prompts the user.
    3. Exits if no key is provided.
    """
    global GEMINI_API_KEY # Use the global variable

    if GEMINI_API_KEY:
        print("Gemini API Key found in environment variable.")
        return GEMINI_API_KEY

    print("GEMINI_API_KEY environment variable not found.")
    GEMINI_API_KEY = input("Please enter your Gemini API Key: ").strip()
    
    if not GEMINI_API_KEY:
        print("Error: Gemini API Key is required to run this demo.")
        print("Please set the GEMINI_API_KEY environment variable or enter it when prompted.")
        exit(1) # Exit if no key is provided
    # print("Gemini API Key obtained via input.") # Optional: confirm how key was obtained
    return GEMINI_API_KEY

def capture_video_frame(rtsp_url):
    """Captures a single frame from the RTSP stream, encodes it as JPEG, then Base64."""
    cap = None  # Initialize cap to None
    try:
        print(f"Attempting to connect to RTSP stream: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            print(f"Error: Could not open RTSP stream at {rtsp_url}")
            return None

        # Set a timeout for reading the frame (e.g., 5 seconds)
        # This requires a loop and checking time, as VideoCapture doesn't have a direct timeout for read()
        # For simplicity in this step, we'll do a direct read.
        # Consider adding a more robust timeout mechanism if needed.
        print("Reading frame from RTSP stream...")
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Error: Failed to read frame from RTSP stream.")
            return None

        print("Frame captured successfully. Encoding to JPEG and Base64...")
        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            print("Error: Failed to encode frame to JPEG.")
            return None

        base64_image = base64.b64encode(encoded_image).decode('utf-8')
        print("Frame encoded to Base64 successfully.")
        return base64_image

    except Exception as e:
        print(f"An error occurred during video capture: {e}")
        return None
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            print("RTSP stream capture released.")

def audio_input_server(host, port):
    """Runs a socket server to receive PCM audio data and updates global variable."""
    # TODO: Implement in Step 3
    global latest_base64_audio_data
    global audio_server_running

    server_socket = None # Initialize server_socket

    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow address reuse
        server_socket.bind((host, port))
        server_socket.listen(1) # Listen for one connection at a time
        print(f"Audio input server listening on {host}:{port}...")

        while audio_server_running:
            conn = None # Initialize conn
            try:
                # Set a timeout for accept, so it doesn't block indefinitely and can check audio_server_running
                server_socket.settimeout(1.0) 
                try:
                    conn, addr = server_socket.accept()
                except socket.timeout:
                    continue # Go back to check audio_server_running

                print(f"Audio client connected from {addr}")
                
                received_data = bytearray() # Use bytearray for efficient concatenation
                # Set a timeout for recv to handle cases where client connects but sends no data
                conn.settimeout(5.0) # Timeout for individual recv calls

                while audio_server_running: # Also check running flag here to stop mid-connection if needed
                    try:
                        chunk = conn.recv(4096) # Receive data in chunks
                        if not chunk:
                            print("Client closed connection or sent no more data.")
                            break # Client closed connection or sent empty data
                        received_data.extend(chunk)
                        # Optional: Add a small delay if expecting very fragmented packets
                        # time.sleep(0.01) 
                    except socket.timeout:
                        # This timeout means no data received in the last 5 seconds on this connection
                        if received_data:
                            print("Socket recv timed out, but some data was received. Assuming client finished sending.")
                        else:
                            print("Socket recv timed out, no data received from client on this connection.")
                        break # Assume client is done sending for this connection
                    except Exception as e:
                        print(f"Error receiving audio data: {e}")
                        break
                
                if received_data:
                    print(f"Received {len(received_data)} bytes of audio data.")
                    latest_base64_audio_data = base64.b64encode(received_data).decode('utf-8')
                    print("Audio data base64 encoded and updated.")
                else:
                    print("No audio data received from this client connection.")

            except Exception as e:
                if audio_server_running: # Only print error if server is supposed to be running
                    print(f"Error in audio server connection/handling: {e}")
            finally:
                if conn:
                    conn.close()
                # If we break from the inner loop due to audio_server_running becoming false
                if not audio_server_running:
                    break 
        
    except Exception as e:
        print(f"Fatal error in audio input server: {e}")
    finally:
        if server_socket:
            server_socket.close()
        print("Audio input server stopped.")


def send_to_gemini(api_key, model_name, text_prompt, base64_image_data, base64_audio_data, audio_mime_type):
    """Sends the multimodal input to the Gemini API and returns the response."""
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    parts = []
    if text_prompt:
        parts.append({"text": text_prompt})
    
    if base64_image_data:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": base64_image_data
            }
        })
        print("Image data will be sent.")

    if base64_audio_data and audio_mime_type:
        parts.append({
            "inline_data": {
                "mime_type": audio_mime_type, # e.g., "audio/L16;rate=8000;channels=1"
                "data": base64_audio_data
            }
        })
        print(f"Audio data with MIME type {audio_mime_type} will be sent.")

    if not parts:
        print("No content (text, image, or audio) to send to Gemini.")
        return None

    request_body = {
        "contents": [{"role": "user", "parts": parts}],
        "safety_settings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
        "generation_config": {
            # "response_mime_type": "application/json" # Default is JSON.
            # If we want Gemini to try and output audio directly:
            # "response_mime_type": "audio/opus" 
            # For this demo, let's stick to the default and parse text/audio from the JSON response.
            # We can add an option later if direct audio output is desired.
        }
    }

    headers = {'Content-Type': 'application/json'}

    print(f"Sending request to Gemini API: {api_url}")
    # print(f"Request body: {json.dumps(request_body, indent=2)}") # Can be very verbose with base64 data

    try:
        response = requests.post(api_url, headers=headers, json=request_body, timeout=60)
        
        print(f"Gemini API response status code: {response.status_code}")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error from Gemini API: {response.status_code}")
            try:
                error_details = response.json()
                print(f"Error details: {json.dumps(error_details, indent=2)}")
            except json.JSONDecodeError:
                print(f"Error details (raw): {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error sending request to Gemini API: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in send_to_gemini: {e}")
        return None

def process_gemini_response(response):
    """Processes the Gemini API response, printing text and saving audio."""
    if response is None:
        print("No response from Gemini to process.")
        return

    try:
        # Basic check for response structure
        if "candidates" not in response:
            print("Error: 'candidates' not found in Gemini response.")
            print(f"Full response for debugging: {json.dumps(response, indent=2)}")
            return

        for candidate_idx, candidate in enumerate(response.get("candidates", [])):
            print(f"--- Candidate {candidate_idx + 1} ---")
            if "content" in candidate and "parts" in candidate["content"]:
                for part_num, part in enumerate(candidate["content"]["parts"]):
                    if "text" in part:
                        print(f"  Text Response (Part {part_num + 1}):\n    {part['text'].replace('\n', '\n    ')}")
                    
                    if "inline_data" in part:
                        inline_data = part["inline_data"]
                        mime_type = inline_data.get("mime_type")
                        data_b64 = inline_data.get("data")

                        if mime_type and data_b64 and mime_type.startswith("audio/"):
                            print(f"  Received audio data (Part {part_num + 1}) with MIME type: {mime_type}")
                            try:
                                audio_bytes = base64.b64decode(data_b64)
                                
                                # Determine file extension
                                extension = mime_type.split('/')[-1].split(';')[0] # e.g., opus, wav, L16
                                if not extension or not extension.replace('-', '').isalnum(): # Basic sanity check, allow hyphens
                                    extension = "bin" # fallback extension

                                # Create a unique filename
                                timestamp = time.strftime("%Y%m%d-%H%M%S")
                                filename = f"gemini_output_audio_{timestamp}_part{part_num + 1}.{extension}"
                                
                                with open(filename, "wb") as f:
                                    f.write(audio_bytes)
                                print(f"  Saved Gemini audio output to: {filename}")

                            except base64.binascii.Error as b64_err:
                                print(f"  Error decoding base64 audio data: {b64_err}")
                            except IOError as io_err:
                                print(f"  Error saving audio file: {io_err}")
                            except Exception as e:
                                print(f"  An unexpected error occurred while processing audio data: {e}")
                        # else:
                        #     print(f"  Received inline_data (Part {part_num + 1}) with non-audio MIME type or missing data: {mime_type}")

            else:
                print(f"  Warning: Candidate {candidate_idx + 1} found without 'content' or 'parts'.")
                # print(f"Problematic candidate: {json.dumps(candidate, indent=2)}")


        # Handle potential top-level errors if not caught by status code check earlier
        # and if no candidates were processed.
        if not response.get("candidates") and response.get("error"):
            print(f"Gemini API returned an error: {json.dumps(response.get('error'), indent=2)}")


    except Exception as e:
        print(f"An error occurred while processing the Gemini response: {e}")
        print(f"Full response for debugging: {json.dumps(response, indent=2)}")

def main():
    """Main function to orchestrate the demo."""
    global latest_base64_audio_data # To allow resetting it
    global audio_server_running   # To control the server thread

    print("\n--- Gemini Multimodal Demo Initializing ---")
    print("This demo supports text, RTSP video frame, and socket-based PCM audio input.")
    
    api_key = get_gemini_api_key()
    # get_gemini_api_key now handles exit if key is not found, so no need to check api_key here.

    audio_thread = None
    try:
        print(f"\nAttempting to start audio input server on {AUDIO_SERVER_HOST}:{AUDIO_SERVER_PORT}...")
        audio_server_running = True # Explicitly set before starting thread
        audio_thread = threading.Thread(target=audio_input_server, args=(AUDIO_SERVER_HOST, AUDIO_SERVER_PORT), daemon=True)
        audio_thread.start()
        
        time.sleep(1) # Give the server a moment to start.
        if not audio_thread.is_alive():
            print("ERROR: Audio server thread failed to start. Please check for port conflicts or other errors.")
            print("Proceeding without audio input capabilities for this session.")
        else:
            print(f"Audio input server thread started successfully.")
            print(f" -> Please send RAW PCM audio data ({AUDIO_SAMPLE_RATE}Hz, {AUDIO_CHANNELS}-channel, {AUDIO_DTYPE} signed integer)")
            print(f"    to TCP socket: {AUDIO_SERVER_HOST_DISPLAY}:{AUDIO_SERVER_PORT}")
            print(f"    Example: nc {AUDIO_SERVER_HOST_DISPLAY} {AUDIO_SERVER_PORT} < my_audio.raw")

        # --- Main Interaction ---
        print("\n--- Ready for User Input ---")
        text_input = input("Enter your text prompt for Gemini (or press Enter to skip text): ").strip()
        
        print("\nAttempting to capture video frame...")
        video_frame_b64 = capture_video_frame(RTSP_URL)
        if video_frame_b64:
            print("Video frame captured.")
        else:
            print("No video frame captured or video input failed.")
        
        # Retrieve the latest audio data captured by the server thread
        # and reset it so it's not used again for a future turn unless new audio comes in.
        current_audio_b64 = latest_base64_audio_data
        latest_base64_audio_data = None # Reset for next potential turn

        if current_audio_b64:
            print(f"Audio data captured ({len(current_audio_b64)} base64 chars).")
            audio_mime = f"audio/L16;rate={AUDIO_SAMPLE_RATE};channels={AUDIO_CHANNELS}"
        else:
            print("No audio data captured for this request.")
            audio_mime = None
            current_audio_b64 = None # Ensure it's explicitly None if no data

        if not text_input and not video_frame_b64 and not current_audio_b64:
            print("\nNo input (text, video, or audio) provided. Exiting demo interaction.")
            return # Exits after finally block

        print("\nSending request to Gemini...")
        gemini_response = send_to_gemini(
            api_key, 
            MODEL_NAME, 
            text_input if text_input else None, # Send None if empty string
            video_frame_b64, 
            current_audio_b64,
            audio_mime
        )
        
        print("\n--- Gemini Response ---")
        if gemini_response:
            process_gemini_response(gemini_response)
        else:
            print("No response received from Gemini or an error occurred.")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down...")
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
    finally:
        if audio_thread is not None: # Check if thread was ever assigned
            print("\nShutting down audio server...")
            audio_server_running = False # Signal the server thread to stop
            if audio_thread.is_alive():
                audio_thread.join(timeout=5) # Wait for the thread to finish
                if audio_thread.is_alive():
                    print("Warning: Audio server thread did not stop cleanly.")
            else:
                print("Audio server thread was not alive or already stopped.")
        print("Demo finished.")

if __name__ == "__main__":
    main()
