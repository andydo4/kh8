#!/usr/bin/env python3
"""
FSRCNN x4 video upscaler with GPU auto-selection (CUDA -> OpenCL -> CPU).
- Reads input.mp4
- Upscales frames using FSRCNN_x4.pb
- Writes output_4x.mp4

Notes:
- For NVIDIA, you need an OpenCV build with CUDA DNN.
- For AMD/Intel, ensure OpenCL is available (ROCm/ICD for AMD, Intel OpenCL runtime, etc.).
- The default pip wheel (opencv-contrib-python) is typically CPU-only for CUDA.
"""

import os
import time
import cv2

# --- 1. Configuration ---
INPUT_VIDEO = "input480.mp4"  # Your source video file
OUTPUT_VIDEO = "output_4x.mp4" # The upscaled file that will be created
MODEL_FILE = "FSRCNN_x4.pb" # The model file you must download
MODEL_NAME = "fsrcnn"
MODEL_SCALE = 4                    # The upscaling factor

# --- 2. Check for Files ---
if not os.path.exists(INPUT_VIDEO):
    print(f"Error: Input video not found at '{INPUT_VIDEO}'. Please name your video 'input.mp4'.")
    raise SystemExit(1)

if not os.path.exists(MODEL_FILE):
    print(f"Error: Model not found at '{MODEL_FILE}'. Please download the 'FSRCNN_x4.pb' file.")
    raise SystemExit(1)

# --- 3. Initialize the Model ---
print(f"Loading model '{MODEL_FILE}'...")
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(MODEL_FILE)
sr.setModel(MODEL_NAME, MODEL_SCALE)

# --- 3a. Try to enable GPU acceleration ---
def enable_gpu_if_possible(superres_obj):
    """
    Tries CUDA first (NVIDIA), then OpenCL (AMD/Intel).
    Falls back to CPU if unavailable.
    Returns a string describing the selected device.
    """
    device = "CPU"
    # Some OpenCV builds expose backend/target setters on DnnSuperResImpl
    has_backend_api = hasattr(superres_obj, "setPreferableBackend") and hasattr(superres_obj, "setPreferableTarget")

    # Always import dnn namespace safely
    try:
        from cv2 import dnn
    except Exception:
        # dnn namespace missing is highly unlikely in contrib, but just in case
        return device

    # Helper to test a backend/target pair
    def try_set(back, target):
        if not has_backend_api:
            return False
        try:
            superres_obj.setPreferableBackend(back)
            superres_obj.setPreferableTarget(target)
            return True
        except Exception:
            return False

    # 1) Try CUDA (NVIDIA)
    # Prefer FP16 target if available for speed, then fall back to full precision
    if has_backend_api:
        # Try FP16 first if symbol exists
        cuda_fp16 = getattr(dnn, "DNN_TARGET_CUDA_FP16", None)
        if try_set(getattr(dnn, "DNN_BACKEND_CUDA", -1), cuda_fp16 if cuda_fp16 is not None else -1):
            device = "CUDA (FP16)" if cuda_fp16 is not None else "CUDA"
            print("Using CUDA backend/target.")
            return device
        # Try full CUDA
        if try_set(getattr(dnn, "DNN_BACKEND_CUDA", -1), getattr(dnn, "DNN_TARGET_CUDA", -2)):
            device = "CUDA"
            print("Using CUDA backend/target.")
            return device

    # 2) Try OpenCL (AMD/Intel)
    # Turn on OpenCL globally
    try:
        cv2.ocl.setUseOpenCL(True)
    except Exception:
        pass

    have_ocl = False
    try:
        have_ocl = cv2.ocl.haveOpenCL()
    except Exception:
        have_ocl = False

    if have_ocl:
        # Prefer FP16 target if available, then regular OpenCL
        ocl_fp16 = getattr(dnn, "DNN_TARGET_OPENCL_FP16", None)
        if try_set(getattr(dnn, "DNN_BACKEND_OPENCV", -1), ocl_fp16 if ocl_fp16 is not None else -1):
            device = "OpenCL (FP16)" if ocl_fp16 is not None else "OpenCL"
            print("Using OpenCL backend/target.")
            return device
        if try_set(getattr(dnn, "DNN_BACKEND_OPENCV", -1), getattr(dnn, "DNN_TARGET_OPENCL", -2)):
            device = "OpenCL"
            print("Using OpenCL backend/target.")
            return device

    # 3) CPU fallback
    # Explicitly set CPU if backend API exists (harmless otherwise)
    if has_backend_api:
        try:
            superres_obj.setPreferableBackend(getattr(dnn, "DNN_BACKEND_OPENCV", 0))
            superres_obj.setPreferableTarget(getattr(dnn, "DNN_TARGET_CPU", 0))
        except Exception:
            pass

    print("GPU acceleration not available; running on CPU.")
    return device

selected_device = enable_gpu_if_possible(sr)
print(f"Model loaded successfully. Compute: {selected_device}")

# --- 4. Open Video Streams and Configure Writer ---
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"Error: Could not open video stream for '{INPUT_VIDEO}'.")
    raise SystemExit(1)

# Get original video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate new dimensions
new_width = original_width * MODEL_SCALE
new_height = original_height * MODEL_SCALE

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for broad compatibility
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (new_width, new_height))

if not writer.isOpened():
    print(f"Error: Could not open video writer for '{OUTPUT_VIDEO}'.")
    cap.release()
    raise SystemExit(1)

print(f"Processing '{INPUT_VIDEO}' ({original_width}x{original_height}) at {fps:.2f} FPS.")
print(f"Outputting to '{OUTPUT_VIDEO}' ({new_width}x{new_height})...")
if selected_device.lower().startswith("cpu"):
    print("Heads up: CPU mode will be VERY SLOW.")

# --- 5. Process the Video (Frame by Frame) ---
start_time = time.time()
processed_frames = 0
last_log = start_time

def fmt_time(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Core upscaling operation
    upscaled_frame = sr.upsample(frame)

    # Write the upscaled frame to the output file
    writer.write(upscaled_frame)
    processed_frames += 1

    # Log progress once per second
    now = time.time()
    if now - last_log >= 1.0:
        elapsed = now - start_time
        fps_proc = processed_frames / elapsed if elapsed > 0 else 0.0
        remaining = (frame_count - processed_frames) / fps_proc if fps_proc > 0 else 0
        print(f"  ... {processed_frames}/{frame_count} frames | {fps_proc:.2f} FPS | ETA {fmt_time(remaining)}")
        last_log = now

# --- 6. Clean Up ---
end_time = time.time()
total_time = end_time - start_time
avg_fps = processed_frames / total_time if total_time > 0 and processed_frames > 0 else 0.0

print("\n--- Done! ---")
print(f"Successfully created '{OUTPUT_VIDEO}'.")
print(f"Processed {processed_frames} frames in {total_time:.2f} seconds.")
print(f"Average processing speed: {avg_fps:.2f} FPS.")

cap.release()
writer.release()
cv2.destroyAllWindows()
