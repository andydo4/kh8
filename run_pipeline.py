# run_pipeline.py
import os, json, struct, math
import cv2, numpy as np
from pathlib import Path

# ---------- Config ----------
VIDEO_PATH = "input.mp4"
OUT_DIR = Path("outputs_fsr2")
TARGET_W = None        # None keeps source size; or set e.g. 1280
USE_RAFT = False       # True if you’ve wired RAFT + GPU
USE_MIDAS_SMALL = False # Small = faster on mac
FLOW_IMPL = "DIS"      # "DIS" or "FARNEBACK" (ignored if USE_RAFT)

# ---------- Device (PyTorch optional) ----------

import torch
if not torch.cuda.is_available():
    raise RuntimeError("ROCm GPU not detected. Check ROCm install and ROCm PyTorch wheels.")
DEVICE = torch.device("cuda")  # HIP under ROCm
AMP = torch.autocast(device_type="cuda", dtype=torch.float16)


# ---------- MiDaS ----------
def load_midas():
    assert torch is not None, "PyTorch required for MiDaS"
    mname = "MiDaS_small" if USE_MIDAS_SMALL else "DPT_Hybrid"
    midas = torch.hub.load('intel-isl/MiDaS', mname).to(DEVICE).eval()
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    tfm = transforms.small_transform if USE_MIDAS_SMALL else transforms.dpt_transform
    return midas, tfm

@torch.inference_mode()
def infer_depth_midas(midas, tfm, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = tfm(rgb)
    if DEVICE != "cpu": inp = inp.to(DEVICE)
    pred = midas(inp.unsqueeze(0))
    depth = pred.squeeze().float().cpu().numpy()
    return depth  # relative (bigger ~ farther)

# ---------- RAFT (optional) ----------
if USE_RAFT:
    from raft import RAFT                         # put your RAFT repo on PYTHONPATH
    from raft_utils import load_raft_weights      # simple helpers you already made
    raft = RAFT().to(DEVICE).eval()
    load_raft_weights(raft, "weights/raft-sintel.pth")

def pad8(t):
    h, w = t.shape[:2]
    nh = (h + 7)//8*8; nw = (w + 7)//8*8
    if (nh, nw) == (h, w): return t, (0,0)
    return cv2.copyMakeBorder(t, 0, nh-h, 0, nw-w, cv2.BORDER_REPLICATE), (nh-h, nw-w)

@torch.inference_mode()
def infer_flow_raft(prev_bgr, curr_bgr):
    # returns (2,H,W) float32 pixels t->t+1
    import torch.nn.functional as F
    def to_tensor(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        t = torch.from_numpy(rgb).permute(2,0,1)[None]
        return t.to(DEVICE)
    t0 = to_tensor(prev_bgr); t1 = to_tensor(curr_bgr)
    a0, pad = pad8(t0[0].permute(1,2,0).cpu().numpy())
    a1, _   = pad8(t1[0].permute(1,2,0).cpu().numpy())
    t0p = torch.from_numpy(a0).permute(2,0,1)[None].to(DEVICE)
    t1p = torch.from_numpy(a1).permute(2,0,1)[None].to(DEVICE)
    out = raft(t0p, t1p, iters=12, test_mode=True)
    flow = out[1][0].permute(1,2,0).float().cpu().numpy()  # HxWx2
    if pad != (0,0):
        h,w = t0.shape[2], t0.shape[3]
        flow = flow[:h, :w, :]
    flow = flow.transpose(2,0,1)  # 2,H,W
    return flow

# ---------- Fast CPU flow ----------
def infer_flow_cpu(prev_bgr, curr_bgr):
    if FLOW_IMPL == "FARNEBACK":
        prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        f = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # HxWx2
        return f.transpose(2,0,1).astype(np.float32)
    else:
        dis = getattr(infer_flow_cpu, "_dis", None)
        if dis is None:
            infer_flow_cpu._dis = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_FAST)
            dis = infer_flow_cpu._dis
        prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        f = dis.calc(prev, curr, None)  # HxWx2
        return f.transpose(2,0,1).astype(np.float32)

# ---------- Helpers ----------
def stabilize_depth(prev_stab, curr_raw):
    # align scale/shift using medians & IQR; EMA blend
    def stats(x):
        med = np.median(x)
        q1,q3 = np.percentile(x, [25,75])
        return med, (q3-q1)
    m0,i0 = stats(prev_stab); m1,i1 = stats(curr_raw)
    s = (i1/(i0+1e-6))
    b = m1 - s*m0
    aligned = s*prev_stab + b
    ema = 0.7
    return ema*prev_stab + (1-ema)*aligned

def write_png_r16(path, img_float):
    # Normalize to [0,1] *per-scene* (monotonic). Clip 1..99% to avoid spikes.
    lo = np.percentile(img_float, 1)
    hi = np.percentile(img_float, 99)
    x = np.clip((img_float - lo)/(hi-lo + 1e-6), 0, 1)
    cv2.imwrite(str(path), (x*65535).astype(np.uint16))

def save_motion_rg16f_bin(path, flow_xy):
    # flow_xy: (2,H,W) float32 in **pixels**, CURRENT->PREVIOUS direction
    # Pack as RG16F (little-endian) raw .bin: [half(u), half(v)] per pixel
    u = flow_xy[0].astype(np.float16)
    v = flow_xy[1].astype(np.float16)
    uv = np.stack([u,v], axis=-1).reshape(-1,2)
    with open(path, "wb") as f:
        f.write(uv.tobytes())

def downscale_to_width(img, target_w):
    if target_w is None: return img
    h, w = img.shape[:2]
    if w <= target_w: return img
    s = target_w / w
    return cv2.resize(img, (target_w, int(round(h*s))), interpolation=cv2.INTER_AREA)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in ["color","depth_r16","motion_rg16f","meta"]:
        (OUT_DIR/d).mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    ok, prev = cap.read()
    if not ok: raise RuntimeError("Could not read first frame")
    prev = downscale_to_width(prev, TARGET_W)

    # Load MiDaS
    midas, tfm = load_midas()

    # First depth
    d_prev_raw = infer_depth_midas(midas, tfm, prev)
    d_prev_stab = d_prev_raw.copy()

    frame_idx = 0
    # write color & depth for first frame
    cv2.imwrite(str(OUT_DIR/"color"/f"{frame_idx:05d}.png"), prev)
    write_png_r16(OUT_DIR/"depth_r16"/f"{frame_idx:05d}.png", d_prev_stab)
    # meta (MV scale in pixels for this resolution)
    H,W = prev.shape[:2]
    with open(OUT_DIR/"meta"/f"{frame_idx:05d}.json","w") as f:
        json.dump({"width": W, "height": H, "MVScaleX": 1.0, "MVScaleY": 1.0}, f)

    while True:
        ok, curr = cap.read()
        if not ok: break
        curr = downscale_to_width(curr, TARGET_W)
        frame_idx += 1
        H,W = curr.shape[:2]

        # FLOW (t->t+1)
        if USE_RAFT:
            flow_fwd = infer_flow_raft(prev, curr)
        else:
            flow_fwd = infer_flow_cpu(prev, curr)

        # Convert to **current->previous** for FSR 2
        flow_curr_to_prev = -flow_fwd

        # DEPTH raw
        d_curr_raw = infer_depth_midas(midas, tfm, curr)
        # DEPTH stabilize
        d_curr_stab = stabilize_depth(d_prev_stab, d_curr_raw)

        # Write outputs
        cv2.imwrite(str(OUT_DIR/"color"/f"{frame_idx:05d}.png"), curr)
        write_png_r16(OUT_DIR/"depth_r16"/f"{frame_idx:05d}.png", d_curr_stab)
        save_motion_rg16f_bin(OUT_DIR/"motion_rg16f"/f"{frame_idx:05d}.bin"),  # RG16F raw

        with open(OUT_DIR/"meta"/f"{frame_idx:05d}.json","w") as f:
            json.dump({"width": W, "height": H, "MVScaleX": 1.0, "MVScaleY": 1.0}, f)

        prev = curr
        d_prev_stab = d_curr_stab

    cap.release()
    print(f"Done. Wrote frames to {OUT_DIR}")

if __name__ == "__main__":
    main()
