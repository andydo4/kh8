#!/usr/bin/env python3

# If needed, download the model with:
# curl -L https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb -o FSRCNN_x2.pb
# curl -L https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb -o FSRCNN_x4.pb

"""
Near-live FSRCNN x4 video upscaler with ROCm/OpenCL
Optimized for minimal latency and maximum throughput using pipeline parallelism.
"""

import os
import time
import cv2
import threading
import queue
import numpy as np
from collections import deque
import warnings
import sys

# Suppress OpenCV warnings about OpenCL
warnings.filterwarnings('ignore')


# Redirect stderr to filter OpenCL warnings
class FilteredStderr:
    def __init__(self):
        self.stderr = sys.stderr
        self.filters = ['ocl4dnn_conv_spatial', 'INVALID_COMPILER_OPTIONS', 'AMD_DEVICE', 'loadTunedConfig']

    def write(self, message):
        if not any(f in message for f in self.filters):
            self.stderr.write(message)

    def flush(self):
        self.stderr.flush()


sys.stderr = FilteredStderr()

# --- AMD/ROCm Workaround ---
# These must be set BEFORE importing cv2
os.environ['OPENCV_OPENCL_DEVICE'] = ':GPU:0'
os.environ['OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES'] = '1'
# Create temp directory for OpenCL cache to suppress warnings
os.environ['OPENCV_OCL4DNN_CONFIG_PATH'] = '/tmp/opencv_ocl4dnn'
os.makedirs('/tmp/opencv_ocl4dnn', exist_ok=True)
# Fix for headless server (no display)
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# --- Configuration ---
INPUT_VIDEO = "input480.mp4"
OUTPUT_VIDEO = "output_2x.mp4"
MODEL_FILE = "FSRCNN_x2.pb"
MODEL_NAME = "fsrcnn"
MODEL_SCALE = 2

# Pipeline tuning (adjust based on your system)
READ_BUFFER_SIZE = 10  # Frames ahead to read
WRITE_BUFFER_SIZE = 10  # Frames to buffer before writing
DISPLAY_PREVIEW = False  # Show live preview window (set False for headless servers)
SKIP_FRAMES_ON_OVERLOAD = False  # Drop frames if GPU can't keep up
BATCH_SIZE = 4  # Process multiple frames at once (experimental)

# --- Validate Files ---
if not os.path.exists(INPUT_VIDEO):
    raise SystemExit(f"Error: Input video '{INPUT_VIDEO}' not found.")
if not os.path.exists(MODEL_FILE):
    raise SystemExit(f"Error: Model '{MODEL_FILE}' not found.\n"
                     f"Download with: curl -L https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/{MODEL_FILE} -o {MODEL_FILE}")


# --- Performance Metrics ---
class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames_read = 0
        self.frames_processed = 0
        self.frames_written = 0
        self.frames_dropped = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0.0

    def update_fps(self):
        with self.lock:
            self.fps_counter += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.current_fps = self.fps_counter / (now - self.last_fps_time)
                self.fps_counter = 0
                self.last_fps_time = now


metrics = Metrics()


# --- Initialize Model with ROCm/OpenCL ---
def setup_model():
    print(f"Loading model '{MODEL_FILE}'...")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(MODEL_FILE)
    sr.setModel(MODEL_NAME, MODEL_SCALE)

    # Enable multi-threading for CPU
    cv2.setNumThreads(os.cpu_count())

    # Always use OpenCV backend
    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    # Check if we should try OpenCL
    try_opencl = True

    # Force OpenCL/ROCm
    try:
        cv2.ocl.setUseOpenCL(True)
        have_ocl = cv2.ocl.haveOpenCL()
    except:
        have_ocl = False

    if have_ocl and try_opencl:
        try:
            # Test OpenCL with a dummy frame
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            test_frame = np.zeros((64, 64, 3), dtype=np.uint8)
            test_result = sr.upsample(test_frame)

            # Validate output
            expected_shape = (64 * MODEL_SCALE, 64 * MODEL_SCALE, 3)
            if test_result.shape == expected_shape:
                print("✓ Using OpenCL (FP32) with ROCm/AMD GPU")
                return sr, "OpenCL"
            else:
                print(f"⚠ OpenCL test failed: wrong output shape {test_result.shape}")
                raise Exception("OpenCL validation failed")

        except Exception as e:
            print(f"⚠ OpenCL failed ({str(e)}), falling back to CPU")
            # Reinitialize for CPU
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(MODEL_FILE)
            sr.setModel(MODEL_NAME, MODEL_SCALE)
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    # Use CPU
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("✓ Using CPU (slower but reliable)")
    print("  Note: OpenCL has known issues with AMD GPUs in OpenCV DNN")
    return sr, "CPU"


sr, device = setup_model()

# --- Video Stream Setup ---
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Error: Could not open '{INPUT_VIDEO}'")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_w, new_h = orig_w * MODEL_SCALE, orig_h * MODEL_SCALE

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (new_w, new_h))
if not video_writer.isOpened():
    raise SystemExit(f"Error: Could not create '{OUTPUT_VIDEO}'")

print(f"\n{'=' * 60}")
print(f"Input:  {orig_w}x{orig_h} @ {fps:.2f} FPS ({frame_count} frames)")
print(f"Output: {new_w}x{new_h}")
print(f"Device: {device}")
print(f"{'=' * 60}\n")

# --- Pipeline Queues ---
frame_queue = queue.Queue(maxsize=READ_BUFFER_SIZE)
result_queue = queue.Queue(maxsize=WRITE_BUFFER_SIZE)
stop_event = threading.Event()


# --- Thread 1: Frame Reader ---
def reader_thread():
    """Continuously reads frames from video source"""
    frame_num = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Non-blocking put with timeout
            frame_queue.put((frame_num, frame), timeout=0.1)
            with metrics.lock:
                metrics.frames_read += 1
            frame_num += 1
        except queue.Full:
            if SKIP_FRAMES_ON_OVERLOAD:
                with metrics.lock:
                    metrics.frames_dropped += 1
                frame_num += 1
            continue

    # Signal end of stream
    frame_queue.put(None)
    print("\n[Reader] Finished reading all frames")


# --- Thread 2: Frame Writer ---
def writer_thread():
    """Writes upscaled frames to output video"""
    expected_frame = 0
    frame_buffer = {}  # Out-of-order frame storage

    while True:
        try:
            result = result_queue.get(timeout=1.0)
            if result is None:  # End signal
                break

            frame_num, upscaled = result
            frame_buffer[frame_num] = upscaled

            # Write frames in order
            while expected_frame in frame_buffer:
                video_writer.write(frame_buffer[expected_frame])

                if DISPLAY_PREVIEW:
                    try:
                        # Display at reduced size for preview
                        preview = cv2.resize(frame_buffer[expected_frame],
                                             (orig_w * 2, orig_h * 2))
                        cv2.imshow('Live Upscaling Preview', preview)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            stop_event.set()
                    except Exception:
                        # Silently disable preview if display fails
                        pass

                del frame_buffer[expected_frame]
                with metrics.lock:
                    metrics.frames_written += 1
                expected_frame += 1

        except queue.Empty:
            if stop_event.is_set():
                break

    print("\n[Writer] Finished writing all frames")


# --- Main Thread: GPU Processing ---
def process_frames():
    """Main processing loop - upscales frames using GPU"""
    print("[Processor] Starting GPU processing...\n")

    frame_batch = []
    frame_nums = []

    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=0.1)
            if item is None:  # End signal
                # Process remaining frames in batch
                if frame_batch:
                    process_batch(frame_batch, frame_nums)
                result_queue.put(None)
                break

            frame_num, frame = item
            frame_batch.append(frame)
            frame_nums.append(frame_num)

            # Process when batch is full or queue is empty
            if len(frame_batch) >= BATCH_SIZE or frame_queue.qsize() == 0:
                process_batch(frame_batch, frame_nums)
                frame_batch = []
                frame_nums = []

        except queue.Empty:
            # Process any accumulated frames
            if frame_batch:
                process_batch(frame_batch, frame_nums)
                frame_batch = []
                frame_nums = []
            continue
        except Exception as e:
            print(f"\n[ERROR] Processing failed: {e}")
            stop_event.set()
            break


def process_batch(frames, frame_nums):
    """Process a batch of frames"""
    start = time.time()

    for i, frame in enumerate(frames):
        try:
            upscaled = sr.upsample(frame)

            # Validate output shape
            expected_h = frame.shape[0] * MODEL_SCALE
            expected_w = frame.shape[1] * MODEL_SCALE
            if upscaled.shape != (expected_h, expected_w, 3):
                raise Exception(f"Invalid output shape: {upscaled.shape}, expected ({expected_h}, {expected_w}, 3)")

            result_queue.put((frame_nums[i], upscaled))

            with metrics.lock:
                metrics.frames_processed += 1

            metrics.update_fps()

        except Exception as e:
            print(f"\n[ERROR] Frame {frame_nums[i]} upscaling failed: {e}")
            stop_event.set()
            return

    gpu_time = (time.time() - start) / len(frames)

    # Status update - show every batch
    with metrics.lock:
        elapsed = time.time() - metrics.start_time
        progress = (metrics.frames_processed / frame_count) * 100
        eta = (frame_count - metrics.frames_processed) / metrics.current_fps if metrics.current_fps > 0 else 0

        # Update more frequently
        if metrics.frames_processed % 10 == 0 or True:
            print(f"Progress: {progress:5.1f}% | "
                  f"FPS: {metrics.current_fps:5.1f} | "
                  f"GPU: {gpu_time * 1000:5.1f}ms | "
                  f"Batch: {len(frames)} | "
                  f"Queue: R={frame_queue.qsize():2d} W={result_queue.qsize():2d} | "
                  f"ETA: {int(eta)}s", end='\r')


# --- Start Pipeline ---
reader = threading.Thread(target=reader_thread, daemon=True)
writer = threading.Thread(target=writer_thread, daemon=True)

reader.start()
writer.start()

try:
    process_frames()  # Main thread does GPU work
except KeyboardInterrupt:
    print("\n\n[INTERRUPTED] Stopping gracefully...")
    stop_event.set()

# Wait for threads to finish
reader.join(timeout=5.0)
writer.join(timeout=5.0)

# Calculate final statistics
elapsed = time.time() - metrics.start_time

# --- Cleanup ---
cap.release()
video_writer.release()
try:
    cv2.destroyAllWindows()
except:
    pass  # Ignore on headless systems

# --- Print Statistics ---
print(f"\n\n{'=' * 60}")
print("PROCESSING COMPLETE")
print(f"{'=' * 60}")
print(f"Total time:       {elapsed:.2f}s")
print(f"Frames read:      {metrics.frames_read}")
print(f"Frames processed: {metrics.frames_processed}")
print(f"Frames written:   {metrics.frames_written}")
if metrics.frames_dropped > 0:
    print(f"Frames dropped:   {metrics.frames_dropped}")
avg_fps = metrics.frames_processed / elapsed if elapsed > 0 else 0
print(f"Average FPS:      {avg_fps:.2f}")
print(f"Speedup factor:   {avg_fps / fps:.2f}x (1.0x = real-time)")
print(f"Output saved:     '{OUTPUT_VIDEO}'")
print(f"{'=' * 60}\n")