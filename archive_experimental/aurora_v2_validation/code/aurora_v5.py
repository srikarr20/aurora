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
# DPI-ALIGNED ACCUMULATION
# =========================
def compute_accumulation(V):

    T = V.shape[2]
    ref = V[:, :, 0]

    A = np.zeros_like(ref)

    for t in range(T):
        frame = V[:, :, t]

        # motion encoding (difference from reference)
        diff = np.abs(frame - ref)

        # normalize like DPI
        if diff.max() > 0:
            diff = diff / diff.max()

        # accumulate
        A += diff

    # average
    A = A / T

    # final normalization (like DPI)
    if A.max() > 0:
        A = A / A.max()

    return A


# =========================
# FEATURE EXTRACTION (FROM ACCUMULATED MAP)
# =========================
def compute_features(A):

    # global strength
    total_energy = float(np.mean(A))

    # active area
    active_area = float((A > 0.2).sum())

    # spatial std (pattern spread)
    spatial_std = float(np.std(A))

    # quadrant features
    h, w = A.shape

    TL = A[:h//2, :w//2]
    TR = A[:h//2, w//2:]
    BL = A[h//2:, :w//2]
    BR = A[h//2:, w//2:]

    r_tl = float(np.mean(TL))
    r_tr = float(np.mean(TR))
    r_bl = float(np.mean(BL))
    r_br = float(np.mean(BR))

    return total_energy, active_area, spatial_std, r_tl, r_tr, r_bl, r_br


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

        # DPI-style accumulation
        A = compute_accumulation(V)

        # extract features
        energy, area, std, tl, tr, bl, br = compute_features(A)

        print(f"{patient} → ENERGY:{energy:.3f} AREA:{area:.0f}")

        results.append([
            patient,
            energy,
            area,
            std,
            tl,
            tr,
            bl,
            br
        ])

    # save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Patient",
            "ENERGY",
            "ACTIVE_AREA",
            "SPATIAL_STD",
            "R_TL", "R_TR", "R_BL", "R_BR"
        ])
        writer.writerows(results)

    print(f"\nSaved: {output_csv}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    process_batch(
        "aurora_v2_validation/data",
        "aurora_v2_validation/outputs/v5_features.csv"
    )
