import os
import numpy as np
import nibabel as nib
import cv2


# =========================
# LOAD MRI
# =========================
def load_data(path):
    V = nib.load(path).get_fdata()
    V = (V - V.min()) / (V.max() - V.min() + 1e-8)

    if len(V.shape) == 4:
        z = V.shape[2] // 2
        V = V[:, :, z, :]

    return V


# =========================
# ACCUMULATION (AURORA)
# =========================
def compute_accumulation(V):

    T = V.shape[2]
    ref = V[:, :, 0]

    A = np.zeros_like(ref)

    for t in range(T):
        frame = V[:, :, t]

        diff = np.abs(frame - ref)

        if diff.max() > 0:
            diff = diff / diff.max()

        A += diff

    A = A / T

    if A.max() > 0:
        A = A / A.max()

    return A


# =========================
# VISUALIZE
# =========================
def visualize(V, A, output_path):

    os.makedirs(output_path, exist_ok=True)

    h, w = V.shape[:2]

    # reference MRI frame
    raw = (V[:, :, 0] * 255).astype(np.uint8)
    raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    # AURORA map
    heat = (A * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    combined = np.hstack([raw, heat])

    # labels
    cv2.putText(combined, "MRI", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.putText(combined, "AURORA (Accumulated Motion)", (w + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    out_path = os.path.join(output_path, "aurora_visual.png")
    cv2.imwrite(out_path, combined)

    print(f"Saved: {out_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    input_path = "aurora_v2_validation/data/patient001/patient001_4d.nii.gz"
    output_path = "aurora_v2_validation/outputs/visual"

    V = load_data(input_path)
    A = compute_accumulation(V)

    visualize(V, A, output_path)
