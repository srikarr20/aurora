import os
import argparse
import numpy as np
import cv2
import nibabel as nib
import csv


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
# MSI
# =========================
def compute_msi(D):
    vals = D[D > 0.15]
    return float(np.percentile(vals, 90)) if len(vals) > 0 else 0.0


# =========================
# CLASSIFICATION
# =========================
def classify(msi):
    if msi > 0.45:
        return "HIGH_MOTION"
    elif msi > 0.30:
        return "NORMAL"
    else:
        return "LOW_CONTRACTION"


# =========================
# VISUALIZATION
# =========================
def visualize(V, D, msi, label, output_path):
    os.makedirs(output_path, exist_ok=True)

    h, w = V.shape[:2]
    video_path = os.path.join(output_path, "aurora_final.mp4")

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

        cv2.putText(combined, f"MSI: {msi:.3f}", (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(combined, f"CLASS: {label}", (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        out.write(combined)

    out.release()


# =========================
# SINGLE PROCESS
# =========================
def process_single(input_path, output_path):
    V = load_data(input_path)
    D = compute_aurora(V)
    msi = compute_msi(D)
    label = classify(msi)

    print(f"{os.path.basename(input_path)} → MSI: {msi:.4f} | {label}")

    visualize(V, D, msi, label, output_path)

    return msi, label


# =========================
# BATCH PROCESS (FIXED)
# =========================
def process_batch(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for patient in sorted(os.listdir(input_dir)):
        p_path = os.path.join(input_dir, patient)

        if not os.path.isdir(p_path):
            continue

        # ONLY pick 4D MRI file
        nii_files = [f for f in os.listdir(p_path) if "_4d.nii.gz" in f]

        if len(nii_files) == 0:
            continue

        nii_path = os.path.join(p_path, nii_files[0])

        out_path = os.path.join(output_dir, patient)

        msi, label = process_single(nii_path, out_path)

        results.append([patient, msi, label])

    # save CSV
    csv_path = os.path.join(output_dir, "results.csv")

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Patient", "MSI", "Label"])
        writer.writerows(results)

    print(f"\nSaved results: {csv_path}")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help="Single file input")
    parser.add_argument("--input_dir", help="Batch directory")
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.input:
        process_single(args.input, args.output)

    elif args.input_dir:
        process_batch(args.input_dir, args.output)

    else:
        print("Provide --input or --input_dir")


if __name__ == "__main__":
    main()
