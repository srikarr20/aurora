import os
import argparse
import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt


# =========================
# LOAD MRI
# =========================
def load_data(path):
    print(f"Loading: {path}")
    V = nib.load(path).get_fdata()

    V = (V - V.min()) / (V.max() - V.min() + 1e-8)

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
# METRICS
# =========================
def compute_msi(D):
    vals = D[D > 0.15]
    return float(np.percentile(vals, 90)) if len(vals) > 0 else 0.0


# =========================
# FREQUENCY ANALYSIS
# =========================
def compute_frequency_features(D):
    # frame energy
    frame_energy = [
        np.mean(f[f > 0.1]) if np.any(f > 0.1) else 0.0
        for f in D
    ]

    frame_energy = np.array(frame_energy)

    # FFT
    fft_vals = np.fft.fft(frame_energy)
    fft_mag = np.abs(fft_vals)

    # ignore DC component
    fft_mag = fft_mag[1:len(fft_mag)//2]

    if len(fft_mag) == 0:
        return 0.0

    # peak sharpness metric
    peak = np.max(fft_mag)
    spread = np.mean(fft_mag)

    ratio = float(peak / (spread + 1e-8))

    return ratio


# =========================
# VISUALIZATION
# =========================
def visualize(V, D, output_path):
    os.makedirs(output_path, exist_ok=True)

    h, w = V.shape[:2]
    video_path = os.path.join(output_path, "aurora_v3.mp4")

    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        5,
        (w * 2, h)
    )

    for t in range(len(D)):
        raw = (V[:, :, t] * 255).astype(np.uint8)
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

        heat = (D[t] * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        combined = np.hstack([raw, heat])
        out.write(combined)

    out.release()


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

    msi = compute_msi(D)
    freq_ratio = compute_frequency_features(D)

    print("\n==== RESULTS ====")
    print(f"MSI: {msi:.4f}")
    print(f"FREQ_RATIO: {freq_ratio:.4f}")

    visualize(V, D, args.output)


if __name__ == "__main__":
    main()
