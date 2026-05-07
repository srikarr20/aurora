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
# SPATIAL FEATURES
# =========================
def compute_features(D):

    # MSI
    vals = D[D > 0.15]
    msi = float(np.percentile(vals, 90)) if len(vals) > 0 else 0.0

    # ENERGY
    frame_energy = np.array([np.mean(f) for f in D])
    energy = float(np.mean(frame_energy))

    # TEMP VAR
    temp_var = float(np.std(frame_energy))

    # ACTIVE AREA
    active_pixels = [(f > 0.15).sum() for f in D]
    active_area = float(np.mean(active_pixels))

    # REGIONAL MSI (4 quadrants)
    h, w = D[0].shape
    regions = []

    for f in D:
        tl = f[:h//2, :w//2]
        tr = f[:h//2, w//2:]
        bl = f[h//2:, :w//2]
        br = f[h//2:, w//2:]

        for r in [tl, tr, bl, br]:
            vals = r[r > 0.15]
            regions.append(np.percentile(vals, 90) if len(vals) > 0 else 0.0)

    regions = np.array(regions).reshape(len(D), 4)
    regional_msi = np.mean(regions, axis=0)

    return msi, energy, temp_var, active_area, regional_msi


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

        msi, energy, temp_var, active_area, regional_msi = compute_features(D)

        print(f"{patient} → MSI:{msi:.3f} AREA:{active_area:.1f}")

        results.append([
            patient,
            msi,
            energy,
            temp_var,
            active_area,
            regional_msi[0],
            regional_msi[1],
            regional_msi[2],
            regional_msi[3]
        ])

    # save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Patient", "MSI", "ENERGY", "TEMP_VAR",
            "ACTIVE_AREA", "R_TL", "R_TR", "R_BL", "R_BR"
        ])
        writer.writerows(results)

    print(f"\nSaved: {output_csv}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    process_batch(
        "aurora_v2_validation/data",
        "aurora_v2_validation/outputs/spatial_features.csv"
    )
