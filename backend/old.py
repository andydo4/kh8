#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
import threading
import queue
import traceback # For error details

# Add ROCm libraries to path (keep this global setup)
rocm_paths = ['/opt/rocm/lib', '/opt/rocm/lib64', '/opt/rocm-5.7.0/lib', '/opt/rocm-6.0.0/lib']
for path in rocm_paths:
    if os.path.exists(path):
        current_ld = os.environ.get('LD_LIBRARY_PATH', '')
        if path not in current_ld:
            os.environ['LD_LIBRARY_PATH'] = f"{path}:{current_ld}"
        # print(f"Added ROCm path: {path}") # Optional print

try:
    import onnxruntime as ort
    # Import SocketIO only if needed, makes testing easier without Flask context
    # from flask_socketio import SocketIO # Assuming Flask-SocketIO is used
except ImportError as e:
    print(f"ERROR: Missing required libraries ({e}). Install with:")
    print("  pip install onnxruntime-rocm Flask-SocketIO opencv-python numpy")
    # sys.exit(1) # Don't exit here, let the caller handle it if needed

# --- Configuration Constants (Can be overridden by function args) ---
READ_BUFFER_SIZE = 20
WRITE_BUFFER_SIZE = 20
BATCH_SIZE = 8
# --- MODIFIED: Use just filenames, search logic will look in root ---
DEFAULT_MODEL_FILENAMES = ["fsrcnn-small-x4.onnx", "FSRCNN_x2_pytorch.onnx", "FSRCNN_x2.onnx"]
# --- End Modification ---

# --- Metrics Class ---
class Metrics:
    # ... (Metrics class remains the same) ...
    def __init__(self):
        self.lock = threading.Lock()
        self.frames_read = 0
        self.frames_processed = 0
        self.frames_written = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0.0

    def update_fps(self, count=1):
        with self.lock:
            self.fps_counter += count
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.current_fps = self.fps_counter / (now - self.last_fps_time) if (now - self.last_fps_time) > 0 else 0
                self.fps_counter = 0
                self.last_fps_time = now

# --- Main Upscaling Function ---
def upscale_video(input_path, output_folder, model_path=None, scale_factor=None, socketio_instance=None, client_sid=None):
    # ... (docstring remains the same) ...
    print(f"\n--- Starting Upscale Process for {os.path.basename(input_path)} ---")
    metrics = Metrics()
    stop_event = threading.Event()

    # --- 1. Validate Inputs & Determine Model/Scale ---
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    os.makedirs(output_folder, exist_ok=True)

    # --- MODIFIED: Path finding logic ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..')) # Go one level up

    actual_model_path = model_path # The path passed from server.py

    # Check if the provided model_path exists (either absolute or relative to where server.py runs from)
    if actual_model_path and os.path.exists(actual_model_path):
        print(f"✓ Using provided model path: {actual_model_path}")
    # If not found directly, check relative to project root (using basename just in case)
    elif actual_model_path:
        potential_path_from_root = os.path.join(project_root, os.path.basename(actual_model_path))
        if os.path.exists(potential_path_from_root):
            actual_model_path = potential_path_from_root
            print(f"✓ Found model relative to project root: {os.path.basename(actual_model_path)}")
        else:
             actual_model_path = None # Mark as not found if checks fail

    # Search defaults in project root if still not found or not provided
    if actual_model_path is None:
        print(f"Model path '{model_path}' not found or not provided. Searching defaults in project root...")
        for mf_name in DEFAULT_MODEL_FILENAMES:
             # Look for the default model name *in the project root directory*
             potential_path = os.path.join(project_root, mf_name)
             if os.path.exists(potential_path):
                 actual_model_path = potential_path
                 print(f"✓ Found default model in root: {os.path.basename(actual_model_path)}")
                 break # Stop searching once found

    if actual_model_path is None or not os.path.exists(actual_model_path):
         # Error message updated to reflect search location
         raise FileNotFoundError(f"ONNX model not found in project root ({project_root}). Searched defaults: {DEFAULT_MODEL_FILENAMES}")
    # --- End Path Finding Modification ---

    # --- Infer scale factor (this part remains the same) ---
    actual_scale_factor = scale_factor
    if actual_scale_factor is None:
        try:
            base = os.path.basename(actual_model_path).lower()
            if "_x2" in base or "-x2" in base: actual_scale_factor = 2
            elif "_x4" in base or "-x4" in base: actual_scale_factor = 4
            else: raise ValueError("Scale factor not provided and couldn't infer from model name.")
            print(f"Inferred scale factor: {actual_scale_factor}x")
        except Exception as e:
            raise ValueError(f"Could not determine scale factor: {e}")
    if actual_scale_factor not in [2, 4]:
        raise ValueError(f"Unsupported scale factor: {actual_scale_factor}. Only 2x or 4x supported.")
    # --- End Scale Factor Inference ---

    # --- 2. Setup ONNX Session ---
    try:
        session, input_name, output_name, device = _setup_onnx_session(actual_model_path) # Pass absolute path
    except Exception as e:
        raise ValueError(f"Failed to setup ONNX session: {e}")

    # --- 3. Setup Video I/O ---
    # ... (VideoCapture, VideoWriter setup remains the same) ...
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_w, new_h = orig_w * actual_scale_factor, orig_h * actual_scale_factor

    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_upscaled_{actual_scale_factor}x_{int(time.time())}{ext}"
    output_path = os.path.join(output_folder, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (new_w, new_h))
    if not video_writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video writer for: {output_path}")

    print(f"Input:   {orig_w}x{orig_h} @ {fps:.2f} FPS ({frame_count} frames)")
    print(f"Output:  {new_w}x{new_h} -> {output_path}")
    print(f"Device:  {device}")
    print(f"Scale:   {actual_scale_factor}x")
    print(f"Model:   {os.path.basename(actual_model_path)}")
    print("-" * 60)

    # --- 4. Setup Pipeline Queues & Threads ---
    # ... (Queue setup remains the same) ...
    frame_queue = queue.Queue(maxsize=READ_BUFFER_SIZE)
    result_queue = queue.Queue(maxsize=WRITE_BUFFER_SIZE)

    # --- Thread Functions ---
    # ... (reader_thread_func remains the same) ...
    def reader_thread_func():
        frame_num = 0
        failed_reads = 0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if failed_reads < 10:
                    failed_reads += 1
                    time.sleep(0.01)
                    continue
                break
            failed_reads = 0
            try:
                frame_queue.put((frame_num, frame), block=True, timeout=0.5)
                with metrics.lock: metrics.frames_read += 1
                frame_num += 1
            except queue.Full:
                if stop_event.is_set(): break
                continue
            except Exception as e:
                 print(f"[Reader Error] {e}")
                 stop_event.set()
                 break
        frame_queue.put(None)
        print(f"[Reader] Finished reading {frame_num} frames.")

    # ... (writer_thread_func remains the same) ...
    def writer_thread_func():
        expected_frame = 0
        frame_buffer = {}
        frames_actually_written = 0
        while True:
            try:
                result = result_queue.get(timeout=1.0)
                if result is None:
                    while expected_frame in frame_buffer:
                        video_writer.write(frame_buffer[expected_frame])
                        del frame_buffer[expected_frame]
                        with metrics.lock: metrics.frames_written += 1
                        frames_actually_written += 1
                        expected_frame += 1
                    break

                frame_num, upscaled = result
                frame_buffer[frame_num] = upscaled

                while expected_frame in frame_buffer:
                    video_writer.write(frame_buffer[expected_frame])
                    del frame_buffer[expected_frame]
                    with metrics.lock: metrics.frames_written += 1
                    frames_actually_written += 1
                    expected_frame += 1

            except queue.Empty:
                if stop_event.is_set() and frame_queue.empty() and result_queue.empty():
                     break
                continue
            except Exception as e:
                print(f"[Writer Error] {e}")
                stop_event.set()
                break
        print(f"[Writer] Finished writing {frames_actually_written} frames.")

    # ... (process_batch remains the same, including SocketIO emit) ...
    def process_batch(frames, frame_nums):
        if not frames: return
        batch_start_time = time.time()
        try:
            # --- Preprocessing: Prepare Y channel for NHWC format ---
            batch_input_y = []
            original_crcb = []
            for frame in frames:
                img_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                img_y = img_ycrcb[:, :, 0]
                original_crcb.append(img_ycrcb[:, :, 1:]) # Keep CrCb

                # Normalize Y
                img_y_norm = img_y.astype(np.float32) / 255.0
                # Add Channel dimension at the end -> (H, W, 1)
                img_y_hwc = np.expand_dims(img_y_norm, axis=-1)
                batch_input_y.append(img_y_hwc)

            # Stack along the batch dimension -> (N, H, W, 1)
            batch_input = np.stack(batch_input_y, axis=0)
            # --- End Preprocessing Change ---

            # Run inference
            outputs = session.run([output_name], {input_name: batch_input})
            # Output shape might now be (N, H_new, W_new, 1)
            upscaled_batch_y = outputs[0]

            # --- Postprocessing: Handle NHWC output ---
            for i, upscaled_y_nhwc in enumerate(upscaled_batch_y):
                # Squeeze the last dimension (Channel) -> (H_new, W_new)
                upscaled_y = np.clip(upscaled_y_nhwc.squeeze(axis=-1) * 255.0, 0, 255).astype(np.uint8)

                # Resize original CrCb channels
                h_new, w_new = upscaled_y.shape
                resized_crcb = cv2.resize(original_crcb[i], (w_new, h_new), interpolation=cv2.INTER_CUBIC)

                # Merge YCrCb and convert back to BGR
                final_ycrcb = cv2.merge((upscaled_y, resized_crcb))
                final_frame_bgr = cv2.cvtColor(final_ycrcb, cv2.COLOR_YCrCb2BGR)
                # --- End Postprocessing Change ---

                result_queue.put((frame_nums[i], final_frame_bgr))

            # ... (rest of process_batch, including progress emission, remains the same) ...
        except Exception as e:
            print(f"\n[ERROR] Batch processing failed: {e}")
            traceback.print_exc()
            stop_event.set()
            if socketio_instance and client_sid:
                 try:
                      socketio_instance.emit('upscale_error', {'message': f"Processing error: {e}"}, to=client_sid)
                 except Exception as emit_e:
                      print(f"WebSocket emit error failed: {emit_e}")

    # ... (processor_loop remains the same) ...
    def processor_loop():
        print("[Processor] Starting...")
        frame_batch = []
        frame_nums = []
        while not stop_event.is_set():
            try:
                item = frame_queue.get(timeout=0.1)
                if item is None:
                    if frame_batch: process_batch(frame_batch, frame_nums)
                    result_queue.put(None)
                    break

                frame_num, frame = item
                frame_batch.append(frame)
                frame_nums.append(frame_num)

                if len(frame_batch) >= BATCH_SIZE:
                    process_batch(frame_batch, frame_nums)
                    frame_batch, frame_nums = [], []

            except queue.Empty:
                reader_alive = any(t.is_alive() for t in threading.enumerate() if t.name == reader_thread.name) # Check if reader is still running
                if frame_batch and frame_queue.empty() and not reader_alive:
                     process_batch(frame_batch, frame_nums)
                     frame_batch, frame_nums = [], []
                continue
            except Exception as e:
                 print(f"[Processor Error] {e}")
                 stop_event.set()
                 result_queue.put(None)
                 break
        print("[Processor] Finishing.")

    # --- 5. Start Threads and Wait ---
    # ... (Thread starting/joining remains the same) ...
    reader_thread = threading.Thread(target=reader_thread_func, name="ReaderThread", daemon=True)
    writer_thread = threading.Thread(target=writer_thread_func, name="WriterThread", daemon=True)
    processor_thread = threading.Thread(target=processor_loop, name="ProcessorThread", daemon=True)

    try:
        metrics.start_time = time.time()
        reader_thread.start()
        writer_thread.start()
        processor_thread.start()

        processor_thread.join()
        writer_thread.join(timeout=60) # Increased timeout

        if processor_thread.is_alive(): print("Warning: Processor thread did not exit cleanly.")
        if writer_thread.is_alive(): print("Warning: Writer thread did not exit cleanly.")

        if stop_event.is_set():
             raise RuntimeError("Upscaling process failed or was interrupted.")

        with metrics.lock:
             # Allow slight mismatch due to potential read errors at end
             if abs(metrics.frames_written - metrics.frames_read) > 5 and metrics.frames_written == 0:
                  print(f"Warning: Frame count mismatch! Read:{metrics.frames_read}, Written:{metrics.frames_written}")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping threads...")
        stop_event.set()
        raise
    except Exception as e:
        print(f"\n[MAIN ERROR] {e}")
        stop_event.set()
        raise
    finally:
        # --- 6. Cleanup ---
        # ... (Cleanup remains the same) ...
        print("Releasing video resources...")
        if cap: cap.release()
        if video_writer: video_writer.release()
        reader_thread.join(timeout=1.0)
        writer_thread.join(timeout=1.0)
        processor_thread.join(timeout=1.0)

    # --- 7. Return Output Path ---
    # ... (Return remains the same) ...
    print(f"--- Upscale Process Finished for {os.path.basename(input_path)} ---")
    return output_path


# --- Helper for ONNX Session Setup (Internal) ---
# ... (_setup_onnx_session remains the same) ...
def _setup_onnx_session(model_file):
    print(f"Initializing ONNX session for: {os.path.basename(model_file)}")
    available_providers = ort.get_available_providers()
    print(f"Available ONNX providers: {available_providers}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    provider_preference = [
        ('ROCMExecutionProvider', {'device_id': 0, 'arena_extend_strategy': 'kNextPowerOfTwo', 'gpu_mem_limit': 16 * 1024 * 1024 * 1024}),
        ('MIGraphXExecutionProvider', {'device_id': 0}),
        ('CUDAExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider'
    ]
    session = None
    used_provider = None

    for provider in provider_preference:
        try:
            provider_name = provider[0] if isinstance(provider, tuple) else provider
            if provider_name in available_providers:
                print(f"Attempting to use provider: {provider_name}")
                providers_list = [provider] if isinstance(provider, tuple) else [provider]
                session = ort.InferenceSession(model_file, sess_options, providers=providers_list)
                used_provider = session.get_providers()[0]
                print(f"✓ Successfully initialized ONNX with {used_provider}")
                break
            else:
                 print(f"Provider not available: {provider_name}")
        except Exception as e:
            print(f"  Failed to initialize with {provider_name}: {e}")
            if provider_name == 'ROCMExecutionProvider' and 'CPUExecutionProvider' in available_providers:
                 print("  Falling back to CPU due to ROCm init failure.")
                 try:
                      session = ort.InferenceSession(model_file, sess_options, providers=['CPUExecutionProvider'])
                      used_provider = session.get_providers()[0]
                      print(f"✓ Successfully initialized ONNX with {used_provider} (fallback)")
                      break
                 except Exception as cpu_e:
                      print(f"  CPU fallback also failed: {cpu_e}")
            continue

    if session is None:
        raise RuntimeError("Could not initialize ONNX Runtime with any available provider.")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name, used_provider

# --- Optional: Add a main block for testing this script directly ---
# --- MODIFIED: Adjust test paths ---
if __name__ == '__main__':
    print("Running old.py directly for testing...")
    # Get project root assuming script is in backend/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Look for test input in project root
    test_input = os.path.join(project_root_dir, "input480.mp4")
    # Output folder in project root
    test_output_folder = os.path.join(project_root_dir, "test_output")
    test_model = None # Let it find default in root
    test_scale = None # Let it infer

    if not os.path.exists(test_input):
         print(f"Test input file '{test_input}' not found in project root. Skipping direct test.")
    else:
         try:
              start_test_time = time.time()
              result_file = upscale_video(test_input, test_output_folder, test_model, test_scale)
              end_test_time = time.time()
              print("\n----- Direct Test Summary -----")
              print(f"Input: {test_input}")
              print(f"Output: {result_file}")
              print(f"Time Taken: {end_test_time - start_test_time:.2f} seconds")
              print("-----------------------------")
         except Exception as e:
              print(f"\n----- Direct Test FAILED -----")
              print(f"Error: {e}")
              traceback.print_exc()
              print("----------------------------")
# --- End Modification ---