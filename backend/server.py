# backend/server.py
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import threading
import time
import traceback
from werkzeug.utils import secure_filename # For safer filenames

# --- Import your refactored upscaling function ---
# Make sure old.py is in the same directory (backend/)
try:
    from old import upscale_video
except ImportError:
    print("ERROR: Could not import upscale_video from old.py. Make sure it's in the backend folder.")
    exit()
# ---

app = Flask(__name__)
CORS(app) # Allow cross-origin requests
app.config['SECRET_KEY'] = 'dev_secret_key!' # Change for production
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 # Allow 1 GB uploads (adjust as needed)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Define Folders (relative to server.py) ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'upscaled_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# ---

# Store basic info about connected clients (use a better structure for production)
clients = {}

def run_upscaling_in_background(filepath, original_name, client_sid, scale_factor):
    """Function to run upscale_video in a separate thread."""
    print(f"[{client_sid}] Starting background task for: {original_name}")
    try:
        # Determine model path based on scale factor (example logic)
        model_map = {
            2: "backend/FSRCNN_x2.onnx", # Or "fsrcnn-small-x2.onnx" if you have it
            4: "backend/fsrcnn-small-x4.onnx"
        }
        model_path_to_use = model_map.get(scale_factor)
        if not model_path_to_use or not os.path.exists(model_path_to_use):
             # Try finding default if specific scale model missing
             print(f"Warning: Model for scale {scale_factor}x not found directly, letting upscale_video search defaults.")
             model_path_to_use = None # Let upscale_video handle default search

        # Call the refactored upscale function
        result_path = upscale_video(
            input_path=filepath,
            output_folder=OUTPUT_FOLDER,
            model_path=model_path_to_use, # Pass determined path or None
            scale_factor=scale_factor,   # Pass the requested scale
            socketio_instance=socketio,  # Pass the SocketIO object
            client_sid=client_sid        # Pass the specific client ID
        )

        # Construct a URL the client can use relative to the server root
        output_filename = os.path.basename(result_path)
        result_url = f"/results/{output_filename}" # e.g., /results/myvideo_upscaled_2x_12345.mp4

        # Notify client on completion
        socketio.emit('upscale_complete', {
            'result_url': result_url,
            'original_name': original_name
        }, to=client_sid)
        print(f"[{client_sid}] Sent upscale_complete. URL: {result_url}")

    except Exception as e:
        print(f"[{client_sid}] Error during background upscaling for {original_name}: {e}")
        traceback.print_exc() # Print full traceback for debugging
        # Notify client of the error
        socketio.emit('upscale_error', {
            'message': f"Processing failed: {str(e)}",
            'original_name': original_name
        }, to=client_sid)
        print(f"[{client_sid}] Sent upscale_error.")
    # No finally block needed here for client cleanup, handled in disconnect

@app.route('/')
def index():
    return "Backend Server is Running!"

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400
    file = request.files['video']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # --- Get client SID ---
    # A more robust way: Client sends SID via WebSocket *after* getting upload confirmation
    # For now, let's assume the client might send it, or we find it.
    client_sid = request.headers.get('X-SocketIO-SID') # Check headers first
    if not client_sid or client_sid not in clients:
        # Fallback: find *a* connected client (only works reliably for single user!)
        client_sid = next(iter(clients), None)
        if client_sid:
             print(f"Warning: Client SID not in headers, guessing SID: {client_sid}")
        else:
             print("Error: Cannot determine client SID for background task.")
             return jsonify({'error': 'Client session not identified for processing'}), 400

    if file:
        # Secure the filename before saving
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        try:
            print(f"[{client_sid}] Saving file to: {filepath}")
            file.save(filepath)
            print(f"[{client_sid}] File saved.")

            # --- Get requested scale factor (default to 2 if not sent) ---
            # You'll need to send this from the frontend later
            scale_factor = int(request.form.get('scale', 2)) # Example: get from form data
            if scale_factor not in [2, 4]:
                return jsonify({'error': 'Invalid scale factor requested'}), 400

            # --- Start background task ---
            thread = threading.Thread(
                target=run_upscaling_in_background,
                args=(filepath, filename, client_sid, scale_factor)
            )
            thread.start()

            # Respond quickly to the client
            return jsonify({
                'message': 'File uploaded, processing started in background',
                'filename': filename,
                'sid_used': client_sid # For debugging
            }), 202 # HTTP 202 Accepted indicates processing started

        except Exception as e:
            print(f"Error during upload/save for {filename}: {e}")
            return jsonify({'error': f'Server error during upload: {str(e)}'}), 500

    return jsonify({'error': 'File processing failed'}), 500

# --- Route to Serve Upscaled Videos ---
@app.route('/results/<path:filename>') # Use path converter for flexibility
def get_result_file(filename):
    # Security: It's CRITICAL to prevent directory traversal attacks here.
    # secure_filename helps, but checking the base directory is better.
    safe_path = os.path.abspath(os.path.join(OUTPUT_FOLDER, filename))
    if not safe_path.startswith(os.path.abspath(OUTPUT_FOLDER)):
        return jsonify({"error": "Invalid file path"}), 400

    print(f"Serving file request: {filename}")
    try:
        return send_from_directory(
            OUTPUT_FOLDER,
            filename,
            as_attachment=False # False = try to display inline (like in <video>)
                               # True = force download prompt
        )
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return jsonify({"error": "File not found"}), 404
# --- End Serve Route ---

@socketio.on('connect')
def handle_connect():
    clients[request.sid] = {} # Store connected client SID
    print(f'Client connected: {request.sid}. Total clients: {len(clients)}')

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in clients:
        del clients[request.sid] # Remove on disconnect
    print(f'Client disconnected: {request.sid}. Total clients: {len(clients)}')

if __name__ == '__main__':
    print(f"Starting server... CWD: {os.getcwd()}")
    print(f"Uploads folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Output folder: {os.path.abspath(OUTPUT_FOLDER)}")
    print(f"Listening on http://0.0.0.0:5000")
    # Use allow_unsafe_werkzeug=True for dev server with SocketIO reloading
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)