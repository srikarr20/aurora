import os
import numpy as np
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
# FEATURE EXTRACTION
# =========================
def compute_features(D):

    # MSI (strong motion)
    vals = D[D > 0.15]
    msi = float(np.percentile(vals, 90)) if len(vals) > 0 else 0.0

    # frame energy (mean per frame)
    frame_energy = [np.mean(f) for f in D]
    frame_energy = np.array(frame_energy)

    # ENERGY (overall motion)
    energy = float(np.mean(frame_energy))

    # TEMPORAL VARIATION
    temp_var = float(np.std(frame_energy))

    return msi, energy, temp_var


# =========================
# BATCH PROCESS
# =========================
def process_batch(input_dir, output_csv):

    results = []

    for patient in sorted(os.listdir(input_dir)):
        p_path = os.path.join(input_dir, patient)

        if not os.path.isdir(p_path):
            continue

        nii_files = [f for f in os.listdir(p_path) if "_4d.nii.gz" in f]

        if len(nii_files) == 0:
            continue

        nii_path = os.path.join(p_path, nii_files[0])

        V = load_data(nii_path)
        D = compute_aurora(V)

        msi, energy, temp_var = compute_features(D)

        print(f"{patient} → MSI:{msi:.3f} ENERGY:{energy:.3f} VAR:{temp_var:.3f}")

        results.append([patient, msi, energy, temp_var])

    # save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Patient", "MSI", "ENERGY", "TEMP_VAR"])
        writer.writerows(results)

    print(f"\nSaved: {output_csv}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    input_dir = "aurora_v2_validation/data"
    output_csv = "aurora_v2_validation/outputs/features.csv"

    process_batch(input_dir, output_csv)
