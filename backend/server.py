from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import os
# Import your existing upscaling functions here
# from upscale_process import run_upscaling_logic

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE3GGKQ44NbwKY/jVFnhGiKzJasUWJAK31JHiINDzTAC server' 
# Allow uploads up to a certain size (e.g., 500MB)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
socketio = SocketIO(app, cors_allowed_origins="*") # Allow connections

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return "Backend Server is Running!" # Simple check

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Save the uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        print(f"Saving file to: {filepath}")
        file.save(filepath)
        print("File saved.")

        # --- Trigger Upscaling ---
        # Ideally, run this in a background thread/task queue
        # so the upload request returns quickly.
        # For simplicity now, we run it directly (will block):
        print("Starting upscale process...")
        try:
            # result_path = run_upscaling_logic(filepath, scale_factor=2) # Your function
            # For testing, let's just pretend:
            result_path = "path/to/upscaled_video.mp4"
            print(f"Upscaling finished. Result: {result_path}")
            # Notify the client via WebSocket that it's done
            socketio.emit('upscale_complete', {'result_url': result_path, 'original_name': file.filename})
            return jsonify({'message': 'File uploaded and processing started', 'filename': file.filename}), 200
        except Exception as e:
            print(f"Error during upscaling: {e}")
            socketio.emit('upscale_error', {'message': str(e), 'original_name': file.filename})
            return jsonify({'error': f'Processing failed: {e}'}), 500
    return jsonify({'error': 'File upload failed'}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)

if __name__ == '__main__':
    print("Starting server on http://0.0.0.0:5000")
    # Use 0.0.0.0 to make it accessible externally
    # The 'allow_unsafe_werkzeug=True' is needed for newer SocketIO/Werkzeug versions for dev server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)