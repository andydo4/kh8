#!/usr/bin/env python3

# If needed, download the model with:
# curl -L https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb -o FSRCNN_x2.pb
# curl -L https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb -o FSRCNN_x4.pb

# !/usr/bin/env python3
"""
ONNX Runtime FSRCNN x2 video upscaler with ROCm acceleration
Properly utilizes AMD GPUs via ROCm ExecutionProvider
"""

import os
import sys
import time
import cv2
import numpy as np
import threading
import queue
from collections import deque

# Check for ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: ONNX Runtime not found. Install with:")
    print("  pip install onnxruntime-rocm")
    sys.exit(1)

# --- Configuration ---
INPUT_VIDEO = "input480.mp4"
OUTPUT_VIDEO = "output_2x_onnx.mp4"
# Try these models in order:
# 1. Your own converted ONNX model
# 2. PyTorch-created ONNX model (untrained but works)
# 3. Fall back to OpenCV
MODEL_FILES = ["FSRCNN_x2.onnx", "FSRCNN_x2_pytorch.onnx"]
MODEL_SCALE = 2

# Pipeline tuning
READ_BUFFER_SIZE = 20
WRITE_BUFFER_SIZE = 20
BATCH_SIZE = 8  # Process multiple frames at once
DISPLAY_PREVIEW = False


# --- Step 1: Convert TensorFlow model to ONNX if needed ---
def convert_pb_to_onnx():
    """Convert FSRCNN_x2.pb to ONNX format"""
    pb_file = "FSRCNN_x2.pb"

    if os.path.exists(MODEL_FILE):
        print(f"✓ ONNX model '{MODEL_FILE}' already exists")
        return True

    if not os.path.exists(pb_file):
        print(f"ERROR: Neither '{MODEL_FILE}' nor '{pb_file}' found")
        print(f"Download the .pb model first:")
        print(f"  curl -L https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb -o FSRCNN_x2.pb")
        return False

    print(f"Converting '{pb_file}' to ONNX format...")
    try:
        import tf2onnx
        import tensorflow as tf
    except ImportError:
        print("ERROR: Need tf2onnx and tensorflow for conversion. Install with:")
        print("  pip install tf2onnx tensorflow")
        return False

    try:
        # Load the TensorFlow model
        print("  Loading TensorFlow model...")
        graph_def = tf.compat.v1.GraphDef()
        with open(pb_file, 'rb') as f:
            graph_def.ParseFromString(f.read())

        # Convert to ONNX
        print("  Converting to ONNX...")
        model_proto, _ = tf2onnx.convert.from_graph_def(
            graph_def,
            input_names=['x:0'],
            output_names=['y:0'],
            opset=13
        )

        # Save ONNX model
        with open(MODEL_FILE, 'wb') as f:
            f.write(model_proto.SerializeToString())

        print(f"✓ Conversion successful: '{MODEL_FILE}'")
        return True

    except Exception as e:
        print(f"ERROR during conversion: {e}")
        print("\nAlternative: Use the OpenCV version or manually convert the model")
        return False


# --- Performance Metrics ---
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


metrics = Metrics()


# --- Initialize ONNX Runtime with ROCm ---
def setup_onnx_session():
    """Create ONNX Runtime session with ROCm acceleration"""

    # Find available model file
    model_file = None
    for mf in MODEL_FILES:
        if os.path.exists(mf):
            model_file = mf
            print(f"✓ Found model: {model_file}")
            break

    if model_file is None:
        print(f"\nNo ONNX model found. Tried: {MODEL_FILES}")
        print("\nQuick solution: Create a PyTorch FSRCNN model")
        print("Run: python3 simple_pytorch_onnx.py")
        print("\nOR download a pretrained model and convert it")
        sys.exit(1)

    print(f"\nInitializing ONNX Runtime session...")

    # Check available providers
    available_providers = ort.get_available_providers()
    print(f"Available providers: {available_providers}")

    # Configure session options
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = os.cpu_count()

    # Try ROCm provider first, then CUDA, then CPU
    provider_preference = [
        ('ROCMExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 16 * 1024 * 1024 * 1024,  # 16GB
            'do_copy_in_default_stream': True,
        }),
        ('CUDAExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider'
    ]

    session = None
    used_provider = None

    for provider in provider_preference:
        try:
            provider_name = provider[0] if isinstance(provider, tuple) else provider
            if provider_name in available_providers:
                providers = [provider] if isinstance(provider, tuple) else [provider]
                session = ort.InferenceSession(model_file, sess_options, providers=providers)
                used_provider = provider_name
                break
        except Exception as e:
            print(f"  Failed to use {provider_name}: {e}")
            continue

    if session is None:
        print("ERROR: Could not initialize ONNX Runtime with any provider")
        sys.exit(1)

    print(f"✓ Using {used_provider}")

    # Get model info
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return session, input_name, output_name, used_provider


# --- Video Setup ---
print(f"\n{'=' * 60}")
print("ONNX Runtime ROCm Video Upscaler")
print(f"{'=' * 60}")

if not os.path.exists(INPUT_VIDEO):
    print(f"ERROR: Input video '{INPUT_VIDEO}' not found")
    sys.exit(1)

session, input_name, output_name, device = setup_onnx_session()

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"ERROR: Could not open '{INPUT_VIDEO}'")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_w, new_h = orig_w * MODEL_SCALE, orig_h * MODEL_SCALE

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (new_w, new_h))
if not video_writer.isOpened():
    print(f"ERROR: Could not create '{OUTPUT_VIDEO}'")
    sys.exit(1)

print(f"\nInput:  {orig_w}x{orig_h} @ {fps:.2f} FPS ({frame_count} frames)")
print(f"Output: {new_w}x{new_h}")
print(f"Device: {device}")
print(f"Batch size: {BATCH_SIZE}")
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
            frame_queue.put((frame_num, frame), timeout=0.1)
            with metrics.lock:
                metrics.frames_read += 1
            frame_num += 1
        except queue.Full:
            continue

    frame_queue.put(None)
    print("\n[Reader] Finished reading all frames")


# --- Thread 2: Frame Writer ---
def writer_thread():
    """Writes upscaled frames to output video"""
    expected_frame = 0
    frame_buffer = {}

    while True:
        try:
            result = result_queue.get(timeout=1.0)
            if result is None:
                break

            frame_num, upscaled = result
            frame_buffer[frame_num] = upscaled

            # Write frames in order
            while expected_frame in frame_buffer:
                video_writer.write(frame_buffer[expected_frame])
                del frame_buffer[expected_frame]
                with metrics.lock:
                    metrics.frames_written += 1
                expected_frame += 1

        except queue.Empty:
            if stop_event.is_set():
                break

    print("\n[Writer] Finished writing all frames")


# --- Main Thread: ONNX Inference ---
def process_frames():
    """Main processing loop using ONNX Runtime"""
    print("[Processor] Starting ONNX inference...\n")

    frame_batch = []
    frame_nums = []

    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=0.1)
            if item is None:
                # Process remaining frames
                if frame_batch:
                    process_batch(frame_batch, frame_nums)
                result_queue.put(None)
                break

            frame_num, frame = item
            frame_batch.append(frame)
            frame_nums.append(frame_num)

            # Process when batch is full
            if len(frame_batch) >= BATCH_SIZE:
                process_batch(frame_batch, frame_nums)
                frame_batch = []
                frame_nums = []

        except queue.Empty:
            if frame_batch and frame_queue.qsize() == 0:
                process_batch(frame_batch, frame_nums)
                frame_batch = []
                frame_nums = []
            continue


def process_batch(frames, frame_nums):
    """Process a batch of frames through ONNX Runtime"""
    start = time.time()

    try:
        # Prepare batch
        # FSRCNN expects input shape: [batch, height, width, channels]
        batch_input = np.stack([frame.astype(np.float32) for frame in frames])

        # Run inference
        outputs = session.run([output_name], {input_name: batch_input})
        upscaled_batch = outputs[0]

        # Convert back to uint8 and queue results
        for i, upscaled in enumerate(upscaled_batch):
            upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)
            result_queue.put((frame_nums[i], upscaled))

        with metrics.lock:
            metrics.frames_processed += len(frames)

        metrics.update_fps(len(frames))

        batch_time = time.time() - start
        fps_batch = len(frames) / batch_time

        # Progress update
        with metrics.lock:
            elapsed = time.time() - metrics.start_time
            progress = (metrics.frames_processed / frame_count) * 100
            eta = (frame_count - metrics.frames_processed) / metrics.current_fps if metrics.current_fps > 0 else 0

            if metrics.frames_processed % 50 == 0 or metrics.frames_processed < 50:
                print(f"Progress: {progress:5.1f}% | "
                      f"FPS: {metrics.current_fps:6.1f} | "
                      f"Batch: {len(frames):2d} frames in {batch_time * 1000:5.1f}ms | "
                      f"Queue: R={frame_queue.qsize():2d} W={result_queue.qsize():2d} | "
                      f"ETA: {int(eta):3d}s", end='\r')

    except Exception as e:
        print(f"\n[ERROR] Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        stop_event.set()


# --- Start Pipeline ---
reader = threading.Thread(target=reader_thread, daemon=True)
writer = threading.Thread(target=writer_thread, daemon=True)

reader.start()
writer.start()

try:
    process_frames()
except KeyboardInterrupt:
    print("\n\n[INTERRUPTED] Stopping...")
    stop_event.set()

# Wait for threads
reader.join(timeout=5.0)
writer.join(timeout=5.0)

# Calculate statistics
elapsed = time.time() - metrics.start_time

# Cleanup
cap.release()
video_writer.release()

# Statistics
print(f"\n\n{'=' * 60}")
print("PROCESSING COMPLETE")
print(f"{'=' * 60}")
print(f"Total time:       {elapsed:.2f}s")
print(f"Frames read:      {metrics.frames_read}")
print(f"Frames processed: {metrics.frames_processed}")
print(f"Frames written:   {metrics.frames_written}")
avg_fps = metrics.frames_processed / elapsed if elapsed > 0 else 0
print(f"Average FPS:      {avg_fps:.2f}")
print(f"Speedup factor:   {avg_fps / fps:.2f}x (1.0x = real-time)")
print(f"Device used:      {device}")
print(f"Output saved:     '{OUTPUT_VIDEO}'")
print(f"{'=' * 60}\n")