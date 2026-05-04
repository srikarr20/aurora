import cv2
import numpy as np
import nibabel as nib
import os

# =========================
# OUTPUT SETUP
# =========================
OUT_DIR = "aurora_results/demo_final"
os.makedirs(OUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(OUT_DIR, "aurora_comparison_clean.mp4")

# =========================
# INPUT CASES (FULL LABELS OK NOW)
# =========================
cases = [
    ("STABLE", "examples/stable/patient029_4d.nii.gz"),
    ("IRREGULAR", "examples/irregular/patient094_4d.nii.gz"),
    ("LOW CONTRACTION", "examples/low_contraction/patient008_4d.nii.gz")
]

# =========================
# LAYOUT CONFIG
# =========================
LABEL_COL_W = 140   # fixed left label column (prevents overlap)
TARGET_H = 256
TARGET_W = 256

header_h = 45
row_h = TARGET_H + header_h

total_w = LABEL_COL_W + (TARGET_W * 2)

# =========================
# LOAD DATA
# =========================
def load_case(path):
    V = nib.load(path).get_fdata()
    V = (V - V.min()) / (V.max() - V.min() + 1e-8)
    return V

data = [(name, load_case(path)) for name, path in cases]

z_list = [V.shape[2] // 2 for _, V in data]
T = min([V.shape[3] for _, V in data])

# =========================
# VIDEO WRITER
# =========================
fps = 5
out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (total_w, row_h * 3)
)

prev_frames = [None, None, None]

print("Generating FINAL clean comparison video...")

# =========================
# MAIN LOOP
# =========================
for t in range(T):
    rows = []

    for i, (label, V) in enumerate(data):
        frame = V[:, :, z_list[i], t]
        frame = cv2.resize(frame, (TARGET_W, TARGET_H))

        base = (frame * 255).astype(np.uint8)
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        left = base.copy()
        right = base.copy()

        # AURORA overlay
        if prev_frames[i] is not None:
            diff = np.abs(frame - prev_frames[i])
            if np.max(diff) > 0:
                diff = diff / np.max(diff)

            heat = cv2.applyColorMap(
                (diff * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )

            right = cv2.addWeighted(right, 0.75, heat, 0.65, 0)

        prev_frames[i] = frame

        combined = np.hstack([left, right])

        # =========================
        # CANVAS
        # =========================
        canvas = np.zeros((row_h, total_w, 3), dtype=np.uint8)

        # header background
        canvas[:header_h, :] = (20, 20, 20)

        # place image panels (shifted right by label column)
        canvas[header_h:, LABEL_COL_W:] = combined

        # =========================
        # LABEL COLUMN (SAFE ZONE)
        # =========================
        TOP_Y = 28

        cv2.putText(canvas, label,
                    (15, TOP_Y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

        # =========================
        # COLUMN LABELS
        # =========================
        mri_center = LABEL_COL_W + (TARGET_W // 2)
        aurora_center = LABEL_COL_W + TARGET_W + (TARGET_W // 2)

        cv2.putText(canvas, "MRI",
                    (mri_center - 25, TOP_Y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 180, 180), 2)

        cv2.putText(canvas, "AURORA",
                    (aurora_center - 45, TOP_Y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

        # divider between MRI and AURORA
        cv2.line(canvas,
                 (LABEL_COL_W + TARGET_W, header_h),
                 (LABEL_COL_W + TARGET_W, header_h + TARGET_H),
                 (80, 80, 80), 1)

        # divider between label column and MRI
        cv2.line(canvas,
                 (LABEL_COL_W, 0),
                 (LABEL_COL_W, row_h),
                 (60, 60, 60), 1)

        rows.append(canvas)

    final_frame = np.vstack(rows)
    out.write(final_frame)

out.release()

print("\n✅ FINAL CLEAN VIDEO READY")
print("Saved to:", OUTPUT_PATH)
