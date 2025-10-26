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
    from flask_socketio import SocketIO # Assuming Flask-SocketIO is used
except ImportError as e:
    print(f"ERROR: Missing required libraries ({e}). Install with:")
    print("  pip install onnxruntime-rocm Flask-SocketIO opencv-python numpy")
    # sys.exit(1) # Don't exit here, let the caller handle it if needed

# --- Configuration Constants (Can be overridden by function args) ---
READ_BUFFER_SIZE = 20
WRITE_BUFFER_SIZE = 20
BATCH_SIZE = 8
DEFAULT_MODEL_FILES = ["backend/fsrcnn-small-x4.onnx", "backend/FSRCNN_x2.onnx"] # Example paths relative to project root

# --- Metrics Class ---
class Metrics:
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
                self.current_fps = self.fps_counter / (now - self.last_fps_time)
                self.fps_counter = 0
                self.last_fps_time = now

# --- Main Upscaling Function ---
def upscale_video(input_path, output_folder, model_path=None, scale_factor=None, socketio_instance=None, client_sid=None):
    """
    Upscales a video using ONNX Runtime with ROCm acceleration.

    Args:
        input_path (str): Path to the input video file.
        output_folder (str): Folder to save the upscaled video.
        model_path (str, optional): Path to the ONNX model file. If None, tries default paths.
        scale_factor (int, optional): The upscale factor (e.g., 2 or 4). If None, tries to infer from model name.
        socketio_instance (SocketIO, optional): Flask-SocketIO instance for progress updates.
        client_sid (str, optional): Client's SocketIO session ID.

    Returns:
        str: The path to the successfully created output video file.

    Raises:
        FileNotFoundError: If input video or model file is not found.
        ValueError: If scale factor cannot be determined or setup fails.
        RuntimeError: If video processing fails.
    """
    print(f"\n--- Starting Upscale Process for {os.path.basename(input_path)} ---")
    metrics = Metrics() # Create metrics instance for this run
    stop_event = threading.Event() # Control threads for this run

    # --- 1. Validate Inputs & Determine Model/Scale ---
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    os.makedirs(output_folder, exist_ok=True)

    actual_model_path = model_path
    if actual_model_path is None or not os.path.exists(actual_model_path):
        print(f"Model path '{model_path}' not provided or not found. Searching defaults...")
        for mf in DEFAULT_MODEL_FILES:
             # Adjust path relative to this script's location if needed
             script_dir = os.path.dirname(__file__)
             potential_path = os.path.join(script_dir, os.path.basename(mf)) # Look in same dir
             if os.path.exists(potential_path):
                 actual_model_path = potential_path
                 print(f"✓ Found default model: {actual_model_path}")
                 break
             # Fallback check relative to project root if backend folder structure exists
             potential_root_path = os.path.abspath(os.path.join(script_dir, '..', mf))
             if os.path.exists(potential_root_path):
                 actual_model_path = potential_root_path
                 print(f"✓ Found default model (relative to root): {actual_model_path}")
                 break

    if actual_model_path is None or not os.path.exists(actual_model_path):
         raise FileNotFoundError(f"ONNX model not found. Check paths: {DEFAULT_MODEL_FILES}")

    # Infer scale factor if not provided
    actual_scale_factor = scale_factor
    if actual_scale_factor is None:
        try:
            # Simple inference based on common naming like "_x2", "_x4"
            base = os.path.basename(actual_model_path).lower()
            if "_x2" in base: actual_scale_factor = 2
            elif "_x4" in base: actual_scale_factor = 4
            elif "-x2" in base: actual_scale_factor = 2
            elif "-x4" in base: actual_scale_factor = 4
            else: raise ValueError("Scale factor not provided and couldn't infer from model name.")
            print(f"Inferred scale factor: {actual_scale_factor}x")
        except Exception as e:
            raise ValueError(f"Could not determine scale factor: {e}")
    if actual_scale_factor not in [2, 4]: # Add supported scales if different
        raise ValueError(f"Unsupported scale factor: {actual_scale_factor}. Only 2x or 4x supported.")

    # --- 2. Setup ONNX Session ---
    try:
        session, input_name, output_name, device = _setup_onnx_session(actual_model_path)
    except Exception as e:
        raise ValueError(f"Failed to setup ONNX session: {e}")

    # --- 3. Setup Video I/O ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_w, new_h = orig_w * actual_scale_factor, orig_h * actual_scale_factor

    # Generate unique output path
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_upscaled_{actual_scale_factor}x_{int(time.time())}{ext}"
    output_path = os.path.join(output_folder, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or other appropriate codec
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
    frame_queue = queue.Queue(maxsize=READ_BUFFER_SIZE)
    result_queue = queue.Queue(maxsize=WRITE_BUFFER_SIZE)

    # --- Thread Functions (defined inside or passed necessary args) ---
    def reader_thread_func():
        frame_num = 0
        failed_reads = 0
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if failed_reads < 10: # Increased retries
                    failed_reads += 1
                    time.sleep(0.01) # Small sleep on fail
                    continue
                # print(f"[Reader] Read failed definitively after frame {frame_num-1}")
                break
            failed_reads = 0
            try:
                # Block slightly if queue is full, prevents busy-waiting
                frame_queue.put((frame_num, frame), block=True, timeout=0.5)
                with metrics.lock: metrics.frames_read += 1
                frame_num += 1
            except queue.Full:
                if stop_event.is_set(): break
                continue # If timeout occurs but not stopped, just retry
            except Exception as e:
                 print(f"[Reader Error] {e}")
                 stop_event.set()
                 break
        frame_queue.put(None) # Signal end
        print(f"[Reader] Finished reading {frame_num} frames.")

    def writer_thread_func():
        expected_frame = 0
        frame_buffer = {}
        frames_actually_written = 0
        while True:
            try:
                result = result_queue.get(timeout=1.0) # Check queue periodically
                if result is None:
                    # Process any remaining buffered frames after processor is done
                    while expected_frame in frame_buffer:
                        video_writer.write(frame_buffer[expected_frame])
                        del frame_buffer[expected_frame]
                        with metrics.lock: metrics.frames_written += 1
                        frames_actually_written += 1
                        expected_frame += 1
                    break # Exit loop

                frame_num, upscaled = result
                frame_buffer[frame_num] = upscaled

                # Write frames in order
                while expected_frame in frame_buffer:
                    video_writer.write(frame_buffer[expected_frame])
                    del frame_buffer[expected_frame]
                    with metrics.lock: metrics.frames_written += 1
                    frames_actually_written += 1
                    expected_frame += 1

            except queue.Empty:
                if stop_event.is_set() and frame_queue.empty() and result_queue.empty():
                     # If stopped and queues are empty, might need to break early
                     break
                continue # If empty but not stopped, just wait longer
            except Exception as e:
                print(f"[Writer Error] {e}")
                stop_event.set()
                break # Exit on error
        print(f"[Writer] Finished writing {frames_actually_written} frames.")


    def process_batch(frames, frame_nums):
        """Processes a batch, emits progress, puts results in queue."""
        if not frames: return
        batch_start_time = time.time()
        try:
            # --- IMPORTANT: Preprocessing ---
            # Your original script processed BGR directly. FSRCNN usually expects Y channel.
            # Sticking to your original script's BGR processing for now.
            # If results look bad, switch to YCrCb processing like in the previous examples.
            batch_input = np.stack([
                 # Original script logic: Normalize BGR and transpose HWC -> CHW
                (frame.astype(np.float33) / 255.0).transpose(2, 0, 1)
                 for frame in frames
            ])

            # Run inference
            outputs = session.run([output_name], {input_name: batch_input})
            upscaled_batch = outputs[0]

            # --- Postprocessing ---
            for i, upscaled_chw in enumerate(upscaled_batch):
                # Transpose CHW -> HWC
                upscaled_hwc = upscaled_chw.transpose(1, 2, 0)
                # Denormalize and convert type
                final_frame = np.clip(upscaled_hwc * 255.0, 0, 255).astype(np.uint8)

                # Put result into the writer queue
                result_queue.put((frame_nums[i], final_frame))

            with metrics.lock:
                metrics.frames_processed += len(frames)
            metrics.update_fps(len(frames)) # Update FPS counter

            # --- Progress Reporting ---
            if socketio_instance and client_sid:
                with metrics.lock:
                    progress = (metrics.frames_processed / frame_count) * 100 if frame_count > 0 else 0
                    current_fps = metrics.current_fps
                try:
                     # Limit emit frequency (e.g., every N frames or every second)
                     # Send every ~20 frames processed or if progress is significant
                    if metrics.frames_processed % 20 == 0 or progress > 99:
                        socketio_instance.emit('upscale_progress',
                                           {'progress': f"{progress:.1f}", 'fps': f"{current_fps:.1f}"},
                                           to=client_sid)
                except Exception as e:
                    print(f"WebSocket emit progress error: {e}") # Non-fatal

        except Exception as e:
            print(f"\n[ERROR] Batch processing failed: {e}")
            traceback.print_exc()
            stop_event.set() # Signal other threads to stop on critical error
            if socketio_instance and client_sid:
                 try:
                      socketio_instance.emit('upscale_error', {'message': f"Processing error: {e}"}, to=client_sid)
                 except Exception as emit_e:
                      print(f"WebSocket emit error failed: {emit_e}")


    # --- Main Processor Loop (similar to your process_frames) ---
    def processor_loop():
        print("[Processor] Starting...")
        frame_batch = []
        frame_nums = []
        while not stop_event.is_set():
            try:
                item = frame_queue.get(timeout=0.1) # Check queue periodically
                if item is None: # End signal from reader
                    if frame_batch: process_batch(frame_batch, frame_nums) # Process remainder
                    result_queue.put(None) # Signal writer to finish
                    break

                frame_num, frame = item
                frame_batch.append(frame)
                frame_nums.append(frame_num)

                if len(frame_batch) >= BATCH_SIZE:
                    process_batch(frame_batch, frame_nums)
                    frame_batch, frame_nums = [], []

            except queue.Empty:
                 # If reader is done (queue empty, None not received yet) and we have a partial batch
                if frame_batch and frame_queue.empty() and not any(t.is_alive() for t in [reader_thread]):
                     process_batch(frame_batch, frame_nums)
                     frame_batch, frame_nums = [], []
                continue # If empty but not stopped, keep waiting
            except Exception as e:
                 print(f"[Processor Error] {e}")
                 stop_event.set()
                 result_queue.put(None) # Signal writer about error
                 break
        print("[Processor] Finishing.")

    # --- 5. Start Threads and Wait ---
    reader_thread = threading.Thread(target=reader_thread_func, daemon=True)
    writer_thread = threading.Thread(target=writer_thread_func, daemon=True)
    processor_thread = threading.Thread(target=processor_loop, daemon=True) # Run processor in thread too

    try:
        metrics.start_time = time.time() # Reset start time
        reader_thread.start()
        writer_thread.start()
        processor_thread.start()

        # Wait for processor to finish (it signals writer)
        processor_thread.join()
        # Wait for writer to finish (it gets signal from processor)
        writer_thread.join(timeout=30) # Add timeout for safety

        # Check if threads are still alive (indicates potential issue)
        if processor_thread.is_alive(): print("Warning: Processor thread did not exit cleanly.")
        if writer_thread.is_alive(): print("Warning: Writer thread did not exit cleanly.")

        # Check if processing was stopped due to an error
        if stop_event.is_set():
             # Check if an error was already emitted, otherwise raise generic
             # (Error handling within process_batch should emit specific errors)
             raise RuntimeError("Upscaling process failed or was interrupted.")

        # Final check on frame counts
        with metrics.lock:
             if metrics.frames_written != metrics.frames_read or metrics.frames_written == 0:
                  print(f"Warning: Frame count mismatch! Read:{metrics.frames_read}, Written:{metrics.frames_written}")
                  # Don't raise error, but log it. Might happen if video ends abruptly.

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping threads...")
        stop_event.set()
        raise # Re-raise interrupt
    except Exception as e:
        print(f"\n[MAIN ERROR] {e}")
        stop_event.set()
        raise # Re-raise error
    finally:
        # --- 6. Cleanup ---
        print("Releasing video resources...")
        if cap: cap.release()
        if video_writer: video_writer.release()
        # Wait briefly for threads to potentially finish after stop signal
        reader_thread.join(timeout=1.0)
        writer_thread.join(timeout=1.0)
        processor_thread.join(timeout=1.0)

    # --- 7. Return Output Path ---
    print(f"--- Upscale Process Finished for {os.path.basename(input_path)} ---")
    return output_path


# --- Helper for ONNX Session Setup (Internal) ---
def _setup_onnx_session(model_file):
    """Loads ONNX model with preferred providers."""
    print(f"Initializing ONNX session for: {os.path.basename(model_file)}")
    available_providers = ort.get_available_providers()
    print(f"Available ONNX providers: {available_providers}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # sess_options.intra_op_num_threads = os.cpu_count() # Manage threads carefully on server

    provider_preference = [
        ('ROCMExecutionProvider', {'device_id': 0, 'arena_extend_strategy': 'kNextPowerOfTwo', 'gpu_mem_limit': 16 * 1024 * 1024 * 1024}),
        ('MIGraphXExecutionProvider', {'device_id': 0}),
        ('CUDAExecutionProvider', {'device_id': 0}), # Keep as fallback if available
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
                used_provider = session.get_providers()[0] # Get the actual used provider
                print(f"✓ Successfully initialized ONNX with {used_provider}")
                break # Stop after successful initialization
            else:
                 print(f"Provider not available: {provider_name}")
        except Exception as e:
            print(f"  Failed to initialize with {provider_name}: {e}")
            # If ROCm fails, explicitly try CPU before giving up entirely?
            if provider_name == 'ROCMExecutionProvider' and 'CPUExecutionProvider' in available_providers:
                 print("  Falling back to CPU due to ROCm init failure.")
                 try:
                      session = ort.InferenceSession(model_file, sess_options, providers=['CPUExecutionProvider'])
                      used_provider = session.get_providers()[0]
                      print(f"✓ Successfully initialized ONNX with {used_provider} (fallback)")
                      break
                 except Exception as cpu_e:
                      print(f"  CPU fallback also failed: {cpu_e}")
            continue # Try next provider in list

    if session is None:
        raise RuntimeError("Could not initialize ONNX Runtime with any available provider.")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name, used_provider

# --- Optional: Add a main block for testing this script directly ---
if __name__ == '__main__':
    print("Running old.py directly for testing...")
    # Example usage (replace with actual paths for testing)
    test_input = "input480.mp4" # Make sure this exists
    test_output_folder = "test_output"
    test_model = None # Let it find default
    test_scale = None # Let it infer

    if not os.path.exists(test_input):
         print(f"Test input file '{test_input}' not found. Skipping direct test.")
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