import os
import argparse
import numpy as np
import cv2
import nibabel as nib


# =========================
# LOAD MRI
# =========================
def load_data(path):
    print(f"Loading: {path}")
    V = nib.load(path).get_fdata()

    # normalize to [0,1]
    V = (V - V.min()) / (V.max() - V.min() + 1e-8)

    # if 4D → take mid slice
    if len(V.shape) == 4:
        z = V.shape[2] // 2
        V = V[:, :, z, :]

    print(f"Shape: {V.shape}")
    return V


# =========================
# AURORA CORE
# =========================
def compute_aurora(V):
    D = []

    for t in range(1, V.shape[2]):
        diff = np.abs(V[:, :, t] - V[:, :, t - 1])

        if np.max(diff) > 0:
            diff = diff / np.max(diff)

        D.append(diff)

    return np.array(D)


# =========================
# METRICS (FINAL V2.2)
# =========================
def compute_metrics(D):
    # MSI → motion strength
    vals = D[D > 0.15]
    msi = float(np.percentile(vals, 90)) if len(vals) > 0 else 0.0

    # VAR → temporal irregularity (correct final)
    frame_energy = [
        np.mean(f[f > 0.1]) if np.any(f > 0.1) else 0.0
        for f in D
    ]

    # frame-to-frame change (captures instability)
    diffs = np.abs(np.diff(frame_energy))
    var = float(np.mean(diffs)) if len(diffs) > 0 else 0.0

    return msi, var


# =========================
# VISUALIZATION
# =========================
def visualize(V, D, msi, var, output_path):
    os.makedirs(output_path, exist_ok=True)

    h, w = V.shape[:2]
    video_path = os.path.join(output_path, "aurora_v2.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 5, (w * 2, h))

    # fallback if codec fails
    if not out.isOpened():
        print("MP4 failed, switching to AVI")
        video_path = os.path.join(output_path, "aurora_v2.avi")
        out = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            5,
            (w * 2, h)
        )

    for t in range(len(D)):
        raw = (V[:, :, t] * 255).astype(np.uint8)
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        heat = (D[t] * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        combined = np.hstack([raw, heat])

        # labels
        cv2.putText(combined, "MRI", (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(combined, "AURORA V2", (w + 20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # metrics overlay
        cv2.putText(combined, f"MSI: {msi:.3f}", (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(combined, f"VAR: {var:.3f}", (w + 20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out.write(combined)

    out.release()
    print(f"Saved video: {video_path}")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    V = load_data(args.input)
    D = compute_aurora(V)
    msi, var = compute_metrics(D)

    print("\n==== RESULTS ====")
    print(f"MSI: {msi:.4f}")
    print(f"VAR: {var:.4f}")

    visualize(V, D, msi, var, args.output)


if __name__ == "__main__":
    main()
