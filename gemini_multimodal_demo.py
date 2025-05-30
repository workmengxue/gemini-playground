"""
Gemini Multimodal Demo with Requests

Purpose:
  This script demonstrates multimodal interaction with Google's Gemini API
  using the `requests` library for HTTP communication. It can send text,
  a video frame captured from an RTSP stream, and fetch raw PCM audio data
  via a TCP client from an external audio server.

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

3. VIDEO_SOURCE_IP & RTSP_URL:
   - VIDEO_SOURCE_IP: IP address of your RTSP video stream server.
     Default: "192.168.66.130" (REPLACE with your camera's IP address)
   - RTSP_URL: The full RTSP URL. It's constructed using VIDEO_SOURCE_IP by default.
     Example: f'rtsp://{VIDEO_SOURCE_IP}/main_ch'
     If your RTSP stream requires username/password, embed them directly in the
     RTSP_URL string or modify the f-string construction.

4. Audio Input (TCP Client):
   - The script acts as a TCP client to fetch audio from your existing audio server.
   - AUDIO_SOURCE_IP: The IP address of your audio server.
     Defaults to VIDEO_SOURCE_IP but can be set independently.
   - AUDIO_SOURCE_PORT: The port your audio server is listening on (e.g., 6791).
   - AUDIO_COMMAND (optional): The command sent to your server after connection (e.g., b"pcm").
   - The script expects to receive RAW PCM audio data from your server.
   - Expected PCM format: Sample Rate = AUDIO_SAMPLE_RATE (e.g., 8000 Hz),
                         Channels = AUDIO_CHANNELS (e.g., 1 for mono),
                         Bit Depth = 16-bit signed integer (AUDIO_DTYPE = 'int16').

Running the Script:
  1. Configure the variables below, especially:
     - `GEMINI_API_KEY` (as env var or be ready to paste it).
     - `VIDEO_SOURCE_IP` (for RTSP video).
     - `AUDIO_SOURCE_IP` and `AUDIO_SOURCE_PORT` if your audio server is different from the video source.
  2. Ensure your audio server (if you're using audio input) is running and accessible.
  3. Run the script: `python gemini_multimodal_demo.py`
  4. The script will:
     - Prompt you to enter a text query for Gemini.
     - Attempt to capture a frame from the RTSP stream.
     - Attempt to connect to your audio server and fetch audio data.
     - Send the combined inputs (text, video frame, audio data) to Gemini.
     - Print Gemini's text response and save any audio output from Gemini to a file.

Troubleshooting:
  - "Error: Could not open RTSP stream": Check `VIDEO_SOURCE_IP` and the RTSP path in `RTSP_URL`.
    Verify camera credentials if needed (you might need to modify `RTSP_URL` structure for username/password).
  - "Timeout connecting to audio source" or "Connection refused": Check `AUDIO_SOURCE_IP` and
    `AUDIO_SOURCE_PORT`. Ensure your audio server is running and accessible from where you
    run this script. Check firewalls.
  - Gemini API errors: Check your API key and ensure the model name is correct.
"""
import os
import requests
import json
import base64
import cv2  # OpenCV for video
import socket
import threading # Keep for main, even if audio server thread is gone
import time

# --- Configuration ---
# 核心配置 (Core Configuration)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # Gemini API密钥,优先从环境变量读取 (Gemini API Key, read from env var first)
MODEL_NAME = "gemini-1.5-flash-latest"             # 使用的Gemini模型名称 (Gemini model name to use)

# 视频配置 (Video Configuration)
VIDEO_SOURCE_IP = "192.168.66.130"                 # 视频流服务器IP地址 (IP address of the video stream server, REPLACE with your camera's IP)
# RTSP_URL_FORMAT = 'rtsp://{username}:{password}@{ip}/your_stream_path' # Optional: if you have a more complex URL structure with auth
RTSP_URL = f'rtsp://{VIDEO_SOURCE_IP}/main_ch'     # RTSP视频流URL (RTSP video stream URL, constructed using VIDEO_SOURCE_IP)
                                                   # If your RTSP stream requires username/password, embed them directly or use a more complex format string.

# 音频输入配置 (TCP客户端从外部服务器获取) - Audio Input Configuration (TCP Client from external server)
AUDIO_SOURCE_IP = VIDEO_SOURCE_IP              # 音频源服务器IP地址 (默认为视频IP,可独立设置) (Audio source server IP, defaults to VIDEO_SOURCE_IP, can be set independently)
AUDIO_SOURCE_PORT = 6791                       # 音频源服务器端口 (Audio source server port)
AUDIO_COMMAND = b"pcm"                         # 连接后发送到音频源的命令 (Command sent to audio source after connection)
AUDIO_BUFFER_SIZE = 4096                       # 接收音频数据的缓冲区大小 (Buffer size for receiving audio data)
MAX_AUDIO_DURATION_SECONDS = 5                 # 单次请求允许接收音频的最大时长（秒） (Max duration to receive audio for a single request)

# 音频格式配置 (Audio Format Configuration - for data received from AUDIO_SOURCE and sent to Gemini)
AUDIO_SAMPLE_RATE = 8000                       # 音频采样率 (例如 8000 Hz) (Audio sample rate, e.g., 8000 Hz)
AUDIO_CHANNELS = 1                             # 音频通道数 (例如 1) (Number of audio channels, e.g., 1 for mono)
AUDIO_DTYPE = 'int16'                          # 音频数据类型 (例如 'int16' -> 16-bit signed integer PCM) (Audio data type)
                                               # 这用于确定发送给Gemini的MIME类型 e.g. audio/L16 (This is used to determine the MIME type for Gemini, e.g., audio/L16)

# --- Global Variables ---
# No global variables needed for audio server/client management anymore.


# --- Function Definitions ---

def get_audio_from_source(host, port, command, buffer_size, max_duration_sec):
    # 获取音频数据的TCP客户端
    # Args:
    #   host (str): 音频源服务器IP地址
    #   port (int): 音频源服务器端口
    #   command (bytes): 连接后发送到音频源的命令 (例如 b"pcm")
    #   buffer_size (int):接收数据的缓冲区大小
    #   max_duration_sec (int): 最大录制/接收音频的时长（秒）
    # Returns:
    #   bytes: 接收到的PCM音频数据，如果出错或没有数据则返回None
    print(f"Attempting to connect to audio source at {host}:{port}...")
    client_socket = None
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(5.0) # Connection timeout
        client_socket.connect((host, port))
        print(f"Connected to audio source. Sending command: {command}")
        client_socket.sendall(command)

        received_data = bytearray()
        start_time = time.time()
        client_socket.settimeout(2.0) # Timeout for individual recv calls

        print("Receiving audio data...")
        while True:
            if (time.time() - start_time) > max_duration_sec:
                print(f"Max audio duration of {max_duration_sec} seconds reached. Stopping audio capture.")
                break
            try:
                chunk = client_socket.recv(buffer_size)
                if not chunk:
                    print("Audio source closed connection or sent no more data.")
                    break  # Server closed connection or sent empty data
                received_data.extend(chunk)
            except socket.timeout:
                # This timeout means no data received in the last X seconds on this connection.
                # If we've already received some data, we can assume the server is done sending for now.
                if received_data:
                    print("Audio receive timed out, data was received. Assuming audio segment complete.")
                else:
                    print("Audio receive timed out, no data received from audio source.")
                break 
            except Exception as e:
                print(f"Error receiving audio data: {e}")
                break
        
        if received_data:
            print(f"Received {len(received_data)} bytes of audio data from source.")
            return bytes(received_data) # Convert bytearray to bytes
        else:
            print("No audio data received from source.")
            return None

    except socket.timeout:
        print(f"Timeout connecting to audio source at {host}:{port}.")
        return None
    except ConnectionRefusedError:
        print(f"Connection refused by audio source at {host}:{port}. Ensure the audio server is running.")
        return None
    except Exception as e:
        print(f"Error connecting to or getting data from audio source: {e}")
        return None
    finally:
        if client_socket:
            try:
                client_socket.shutdown(socket.SHUT_RDWR) # Gracefully shutdown
            except OSError:
                pass # Ignore if already closed
            client_socket.close()
            print("Audio source client socket closed.")

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
        }
    }

    headers = {'Content-Type': 'application/json'}

    print(f"Sending request to Gemini API: {api_url}")

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
        if "candidates" not in response:
            print("Error: 'candidates' not found in Gemini response.")
            print(f"Full response for debugging: {json.dumps(response, indent=2)}")
            return

        for candidate_idx, candidate in enumerate(response.get("candidates", [])):
            print(f"--- Candidate {candidate_idx + 1} ---")
            if "content" in candidate and "parts" in candidate["content"]:
                for part_num, part in enumerate(candidate["content"]["parts"]):
                    if "text" in part:
                        text_content = part['text']
                        formatted_text = text_content.replace('\n', '\n    ')
                        print(f"  Text Response (Part {part_num + 1}):\n    {formatted_text}")
                    
                    if "inline_data" in part:
                        inline_data = part["inline_data"]
                        mime_type = inline_data.get("mime_type")
                        data_b64 = inline_data.get("data")

                        if mime_type and data_b64 and mime_type.startswith("audio/"):
                            print(f"  Received audio data (Part {part_num + 1}) with MIME type: {mime_type}")
                            try:
                                audio_bytes = base64.b64decode(data_b64)
                                timestamp = time.strftime("%Y%m%d-%H%M%S")
                                extension = mime_type.split('/')[-1].split(';')[0]
                                if not extension or not extension.replace('-', '').isalnum():
                                    extension = "bin"
                                filename = f"gemini_output_audio_{timestamp}_part{part_num + 1}.{extension}"
                                with open(filename, "wb") as f:
                                    f.write(audio_bytes)
                                print(f"  Saved Gemini audio output to: {filename}")
                            except Exception as e:
                                print(f"  Error processing/saving audio data: {e}")
            else:
                print(f"  Warning: Candidate {candidate_idx + 1} found without 'content' or 'parts'.")

        if not response.get("candidates") and response.get("error"):
            print(f"Gemini API returned an error: {json.dumps(response.get('error'), indent=2)}")

    except Exception as e:
        print(f"An error occurred while processing the Gemini response: {e}")
        print(f"Full response for debugging: {json.dumps(response, indent=2)}")

def main():
    """Main function to orchestrate the demo."""
    print("\n--- Gemini Multimodal Demo Initializing ---")
    print("This demo supports text, RTSP video frame, and audio input via TCP client.") 
    
    api_key = get_gemini_api_key()
    
    try:
        print("\n--- Ready for User Input ---")
        text_input = input("Enter your text prompt for Gemini (or press Enter to skip text): ").strip()
        
        print("\nAttempting to capture video frame...")
        video_frame_b64 = capture_video_frame(RTSP_URL) # RTSP_URL is now from global config
        if video_frame_b64:
            print("Video frame captured.")
        else:
            print("No video frame captured or video input failed.")
        
        print("\nAttempting to fetch audio from TCP source...")
        raw_audio_data = get_audio_from_source(
            AUDIO_SOURCE_IP, 
            AUDIO_SOURCE_PORT,
            AUDIO_COMMAND,
            AUDIO_BUFFER_SIZE,
            MAX_AUDIO_DURATION_SECONDS
        )

        current_audio_b64 = None
        audio_mime_type = None 

        if raw_audio_data:
            print(f"Audio data fetched successfully ({len(raw_audio_data)} bytes). Encoding to Base64...")
            current_audio_b64 = base64.b64encode(raw_audio_data).decode('utf-8')
            audio_mime_type = f"audio/L16;rate={AUDIO_SAMPLE_RATE};channels={AUDIO_CHANNELS}" 
            print("Audio Base64 encoded.")
        else:
            print("No audio data fetched from source or an error occurred.")

        if not text_input and not video_frame_b64 and not current_audio_b64:
            print("\nNo input (text, video, or audio) provided. Exiting demo interaction.")
            return 

        print("\nSending request to Gemini...")
        gemini_response = send_to_gemini(
            api_key, 
            MODEL_NAME, 
            text_input if text_input else None,
            video_frame_b64, 
            current_audio_b64, 
            audio_mime_type    
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
        print("Demo finished.")

if __name__ == "__main__":
    main()
