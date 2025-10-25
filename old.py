#!/usr/bin/env python3

# If needed, download the model with:
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
OUTPUT_VIDEO = "output_4x.mp4"
MODEL_FILE = "FSRCNN_x4.pb"
MODEL_NAME = "fsrcnn"
MODEL_SCALE = 4

# Pipeline tuning (adjust based on your system)
READ_BUFFER_SIZE = 10      # Frames ahead to read
WRITE_BUFFER_SIZE = 10     # Frames to buffer before writing
DISPLAY_PREVIEW = False    # Show live preview window (set False for headless servers)
SKIP_FRAMES_ON_OVERLOAD = False  # Drop frames if GPU can't keep up

# --- Validate Files ---
if not os.path.exists(INPUT_VIDEO):
    raise SystemExit(f"Error: Input video '{INPUT_VIDEO}' not found.")
if not os.path.exists(MODEL_FILE):
    raise SystemExit(f"Error: Model '{MODEL_FILE}' not found. Download FSRCNN_x4.pb")

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

    # Force OpenCL/ROCm
    try:
        cv2.ocl.setUseOpenCL(True)
    except:
        pass

    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    have_ocl = cv2.ocl.haveOpenCL()
    if have_ocl:
        # Try standard OpenCL first (FP16 causes issues on AMD)
        try:
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            print("✓ Using OpenCL (FP32) with ROCm/AMD GPU")
            return sr, "OpenCL"
        except Exception as e:
            print(f"⚠ OpenCL setup warning: {e}")
            pass

    # Fallback to CPU
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("⚠ OpenCL unavailable, using CPU (will be slow)")
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

print(f"\n{'='*60}")
print(f"Input:  {orig_w}x{orig_h} @ {fps:.2f} FPS ({frame_count} frames)")
print(f"Output: {new_w}x{new_h}")
print(f"Device: {device}")
print(f"{'='*60}\n")

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

    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=0.1)
            if item is None:  # End signal
                result_queue.put(None)
                break

            frame_num, frame = item

            # GPU upscaling (this is the bottleneck)
            start = time.time()
            upscaled = sr.upsample(frame)
            gpu_time = time.time() - start

            result_queue.put((frame_num, upscaled))

            with metrics.lock:
                metrics.frames_processed += 1

            metrics.update_fps()

            # Status update
            if metrics.frames_processed % 30 == 0:
                with metrics.lock:
                    elapsed = time.time() - metrics.start_time
                    progress = (metrics.frames_processed / frame_count) * 100
                    eta = (frame_count - metrics.frames_processed) / metrics.current_fps if metrics.current_fps > 0 else 0

                    print(f"Progress: {progress:5.1f}% | "
                          f"FPS: {metrics.current_fps:5.1f} | "
                          f"GPU: {gpu_time*1000:5.1f}ms | "
                          f"Queue: R={frame_queue.qsize():2d} W={result_queue.qsize():2d} | "
                          f"ETA: {int(eta)}s", end='\r')

        except queue.Empty:
            continue
        except Exception as e:
            print(f"\n[ERROR] Processing failed: {e}")
            stop_event.set()
            break

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

# --- Cleanup & Statistics ---
cap.release()
video_writer.release()
cv2.destroyAllWindows()

elapsed = time.time() - metrics.start_time
print(f"\n\n{'='*60}")
print("PROCESSING COMPLETE")
print(f"{'='*60}")
print(f"Total time:       {elapsed:.2f}s")
print(f"Frames read:      {metrics.frames_read}")
print(f"Frames processed: {metrics.frames_processed}")
print(f"Frames written:   {metrics.frames_written}")
if metrics.frames_dropped > 0:
    print(f"Frames dropped:   {metrics.frames_dropped}")
print(f"Average FPS:      {metrics.frames_processed / elapsed:.2f}")
print(f"Output saved:     '{OUTPUT_VIDEO}'")
print(f"{'='*60}\n")