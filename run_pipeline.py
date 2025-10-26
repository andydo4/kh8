# run_pipeline.py - Highly optimized multithreaded processing
# NEW PUSH
import os, json, struct, math
import cv2, numpy as np
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from queue import Queue
import threading

# ---------- Config ----------
VIDEO_PATH = "input480.mp4"
OUT_DIR = Path("outputs_fsr2")
TARGET_W = None  # None keeps source size
FLOW_IMPL = "FARNEBACK"  # Use Farneback (always available)
NUM_WORKERS = min(32, mp.cpu_count() * 2)  # Use more threads for I/O bound work

# Validate input file
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video file '{VIDEO_PATH}' not found")

print(f"Using {NUM_WORKERS} parallel workers")


# ---------- Fast Gradient-Based Depth ----------
def estimate_depth_simple(bgr):
    """Fast depth estimation using image gradients"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_magnitude = cv2.GaussianBlur(gradient_magnitude, (15, 15), 0)
    depth = gradient_magnitude.astype(np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth = 1.0 - depth
    depth = cv2.bilateralFilter(depth, 9, 75, 75)
    return depth


# ---------- Optical Flow ----------
def compute_optical_flow(prev_gray, curr_gray):
    """Compute optical flow using Farneback (fast and always available)"""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5,  # Pyramid scale
        levels=3,  # Number of pyramid levels
        winsize=15,  # Window size
        iterations=3,  # Iterations at each level
        poly_n=5,  # Size of pixel neighborhood
        poly_sigma=1.2,  # Gaussian std for smoothing
        flags=0
    )
    return flow.transpose(2, 0, 1).astype(np.float32)


# ---------- Helpers ----------
def stabilize_depth(prev_stab, curr_raw):
    """Align depth scale/shift using statistics"""

    def stats(x):
        med = np.median(x)
        q1, q3 = np.percentile(x, [25, 75])
        return med, (q3 - q1)

    m0, i0 = stats(prev_stab)
    m1, i1 = stats(curr_raw)
    s = (i1 / (i0 + 1e-6))
    b = m1 - s * m0
    aligned = s * prev_stab + b
    ema = 0.7
    return ema * prev_stab + (1 - ema) * aligned


def write_png_r16(path, img_float):
    """Write 16-bit depth PNG"""
    lo = np.percentile(img_float, 1)
    hi = np.percentile(img_float, 99)
    x = np.clip((img_float - lo) / (hi - lo + 1e-6), 0, 1)
    cv2.imwrite(str(path), (x * 65535).astype(np.uint16))


def save_motion_rg16f_bin(path, flow_xy):
    """Save motion vectors as RG16F binary"""
    u = flow_xy[0].astype(np.float16)
    v = flow_xy[1].astype(np.float16)
    uv = np.stack([u, v], axis=-1).reshape(-1, 2)
    with open(path, "wb") as f:
        f.write(uv.tobytes())


def downscale_to_width(img, target_w):
    if target_w is None: return img
    h, w = img.shape[:2]
    if w <= target_w: return img
    s = target_w / w
    return cv2.resize(img, (target_w, int(round(h * s))), interpolation=cv2.INTER_AREA)


# ---------- Pipeline Processing ----------
def process_depth_and_flow(prev_bgr, curr_bgr):
    """Process depth and flow for a frame pair (CPU intensive)"""
    # Depth estimation
    d_curr_raw = estimate_depth_simple(curr_bgr)

    # Optical flow
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    flow_fwd = compute_optical_flow(prev_gray, curr_gray)
    flow_curr_to_prev = -flow_fwd

    return d_curr_raw, flow_curr_to_prev


def write_outputs(frame_idx, frame, depth, flow, out_dir):
    """Write all outputs for a frame (I/O intensive)"""
    H, W = frame.shape[:2]
    cv2.imwrite(str(out_dir / "color" / f"{frame_idx:05d}.png"), frame)
    write_png_r16(out_dir / "depth_r16" / f"{frame_idx:05d}.png", depth)
    save_motion_rg16f_bin(out_dir / "motion_rg16f" / f"{frame_idx:05d}.bin", flow)
    with open(out_dir / "meta" / f"{frame_idx:05d}.json", "w") as f:
        json.dump({"width": W, "height": H, "MVScaleX": 1.0, "MVScaleY": 1.0}, f)


def main():
    # Setup output directories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in ["color", "depth_r16", "motion_rg16f", "meta"]:
        (OUT_DIR / d).mkdir(exist_ok=True, parents=True)

    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {VIDEO_PATH}")

    # Read all frames into memory
    print("Reading video frames...")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(downscale_to_width(frame, TARGET_W))
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        raise RuntimeError("No frames read from video")

    H, W = frames[0].shape[:2]
    print(f"Video: {W}x{H}, {total_frames} frames")
    print(f"Processing with {NUM_WORKERS} workers...\n")

    # Process first frame
    d_prev_raw = estimate_depth_simple(frames[0])
    d_prev_stab = d_prev_raw.copy()

    write_outputs(0, frames[0], d_prev_stab, np.zeros((2, H, W), dtype=np.float32), OUT_DIR)

    start_time = time.time()

    # Parallel processing with pipeline
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks at once
        future_to_idx = {}
        for i in range(1, total_frames):
            future = executor.submit(process_depth_and_flow, frames[i - 1], frames[i])
            future_to_idx[future] = i

        # Process results as they complete
        completed = 0
        results = {}  # Store results to process in order
        prev_depth_stab = d_prev_stab

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            d_raw, flow = future.result()
            results[idx] = (d_raw, flow)

            # Process in sequential order for depth stabilization
            while (completed + 1) in results:
                next_idx = completed + 1
                d_raw, flow = results.pop(next_idx)

                # Stabilize depth
                d_stab = stabilize_depth(prev_depth_stab, d_raw)
                prev_depth_stab = d_stab

                # Write outputs (can be done in background but keeping simple)
                write_outputs(next_idx, frames[next_idx], d_stab, flow, OUT_DIR)

                completed = next_idx

                # Progress
                elapsed = time.time() - start_time
                fps_processing = completed / elapsed if elapsed > 0 else 0
                eta = (total_frames - completed) / fps_processing if fps_processing > 0 else 0
                print(f"Frame {completed}/{total_frames} | {fps_processing:.1f} FPS | ETA: {int(eta)}s", end='\r')

    elapsed = time.time() - start_time
    print(f"\n\n✓ Done! Processed {total_frames} frames in {elapsed:.1f}s ({total_frames / elapsed:.1f} FPS)")
    print(f"  Output: {OUT_DIR}")
    print(f"  Speedup: {(total_frames / elapsed) / 30.0:.1f}x realtime (assuming 30 FPS source)")


if __name__ == "__main__":
    main()