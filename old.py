#!/usr/bin/env python3
"""
FSRCNN x4 video upscaler forcing AMD/OpenCL (no CUDA).
- Reads input480.mp4
- Upscales frames using FSRCNN_x4.pb
- Writes output_4x.mp4

It will:
1) Enable OpenCL and try OPENCL_FP16, then OPENCL
2) If OpenCL isn’t available, fall back to CPU
3) Never attempt CUDA
"""

import os
import time
import cv2

# --- 1. Configuration ---
INPUT_VIDEO = "input480.mp4"    # Your source video file
OUTPUT_VIDEO = "output_4x.mp4"  # The upscaled file that will be created
MODEL_FILE = "FSRCNN_x4.pb"     # The model file you must download
MODEL_NAME = "fsrcnn"
MODEL_SCALE = 4                 # The upscaling factor

# --- 2. Check for Files ---
if not os.path.exists(INPUT_VIDEO):
    print(f"Error: Input video not found at '{INPUT_VIDEO}'.")
    raise SystemExit(1)

if not os.path.exists(MODEL_FILE):
    print(f"Error: Model not found at '{MODEL_FILE}'. Please download 'FSRCNN_x4.pb'.")
    raise SystemExit(1)

# --- 3. Initialize the Model ---
print(f"Loading model '{MODEL_FILE}'...")
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(MODEL_FILE)
sr.setModel(MODEL_NAME, MODEL_SCALE)

# --- 3a. Force AMD/OpenCL (never CUDA) ---
def force_amd_opencl(superres_obj):
    """
    Try OpenCL FP16, then OpenCL. If neither is available, use CPU.
    Never attempts CUDA.
    Returns a string describing the selected device.
    """
    # Make sure OpenCL is enabled globally
    try:
        cv2.ocl.setUseOpenCL(True)
    except Exception:
        pass

    have_ocl = False
    try:
        have_ocl = cv2.ocl.haveOpenCL()
    except Exception:
        have_ocl = False

    # Always use OpenCV backend (not CUDA backend)
    superres_obj.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    if have_ocl:
        # Prefer FP16 if available
        target_set = False
        if hasattr(cv2.dnn, "DNN_TARGET_OPENCL_FP16"):
            try:
                superres_obj.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
                print("Using OpenCL (FP16) target.")
                return "OpenCL (FP16)"
            except Exception:
                target_set = False

        # Fall back to standard OpenCL
        try:
            superres_obj.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            print("Using OpenCL target.")
            return "OpenCL"
        except Exception:
            pass

    # Final fallback: CPU
    superres_obj.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("OpenCL not available; using CPU target.")
    return "CPU"

selected_device = force_amd_opencl(sr)
print(f"Model loaded successfully. Compute: {selected_device}")

# --- 4. Open Video Streams and Configure Writer ---
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"Error: Could not open video stream for '{INPUT_VIDEO}'.")
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

new_width = original_width * MODEL_SCALE
new_height = original_height * MODEL_SCALE

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # broad compatibility
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (new_width, new_height))
if not writer.isOpened():
    print(f"Error: Could not open video writer for '{OUTPUT_VIDEO}'.")
    cap.release()
    raise SystemExit(1)

print(f"Processing '{INPUT_VIDEO}' ({original_width}x{original_height}) at {fps:.2f} FPS.")
print(f"Outputting to '{OUTPUT_VIDEO}' ({new_width}x{new_height})...")
if selected_device == "CPU":
    print("Heads up: CPU mode will be very slow.")

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

    upscaled_frame = sr.upsample(frame)
    writer.write(upscaled_frame)
    processed_frames += 1

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

# If needed, download the model with:
# curl -L https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb -o FSRCNN_x4.pb
