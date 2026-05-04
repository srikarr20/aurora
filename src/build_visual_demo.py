import cv2
import numpy as np
import nibabel as nib
import os
import argparse

# =========================
# INPUT ARGUMENT
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

INPUT = args.input

# =========================
# OUTPUT SETUP
# =========================
OUT_DIR = "aurora_results/demo_final"
os.makedirs(OUT_DIR, exist_ok=True)

name = os.path.basename(INPUT).replace(".nii.gz", "")
VIDEO_PATH = os.path.join(OUT_DIR, f"{name}_aurora.mp4")

print("Loading MRI:", INPUT)

# =========================
# LOAD MRI
# =========================
V = nib.load(INPUT).get_fdata()

z = V.shape[2] // 2
T = V.shape[3]

V = (V - V.min()) / (V.max() - V.min() + 1e-8)

h, w = V.shape[0], V.shape[1]

# =========================
# VIDEO SETUP
# =========================
header_h = 40
fps = 5

out = cv2.VideoWriter(
    VIDEO_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w * 2, h + header_h)
)

prev = None

print("Generating clean demo...")

for t in range(T):
    frame = V[:, :, z, t]

    # base grayscale
    base = (frame * 255).astype(np.uint8)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    # =========================
    # LEFT (RAW MRI)
    # =========================
    left = base.copy()

    # =========================
    # RIGHT (AURORA)
    # =========================
    right = base.copy()

    if prev is not None:
        diff = np.abs(frame - prev)
        if np.max(diff) > 0:
            diff = diff / np.max(diff)

        heat = cv2.applyColorMap(
            (diff * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        right = cv2.addWeighted(right, 0.75, heat, 0.65, 0)

    prev = frame

    # =========================
    # ADD HEADER BAR
    # =========================
    combined = np.hstack([left, right])
    canvas = np.zeros((h + header_h, w * 2, 3), dtype=np.uint8)

    # dark header
    canvas[:header_h, :] = (20, 20, 20)
    canvas[header_h:, :] = combined

    # labels
    cv2.putText(canvas, "RAW MRI",
                (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (200, 200, 200), 2)

    cv2.putText(canvas, "AURORA MOTION MAP",
                (w + 20, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    out.write(canvas)

out.release()

print("\n✅ CLEAN VIDEO CREATED")
print("Saved to:", VIDEO_PATH)
