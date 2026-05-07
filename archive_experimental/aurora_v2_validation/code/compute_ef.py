import os
import numpy as np
import nibabel as nib
import csv


# =========================
# LOAD MASK VOLUME
# =========================
def compute_volume(mask_path):
    mask = nib.load(mask_path).get_fdata()
    return np.sum(mask > 0)


# =========================
# COMPUTE EF PER PATIENT
# =========================
def compute_ef(patient_path):
    files = os.listdir(patient_path)

    gt_files = [f for f in files if "_gt.nii.gz" in f]

    if len(gt_files) != 2:
        return None

    gt_paths = [os.path.join(patient_path, f) for f in gt_files]

    volumes = [compute_volume(p) for p in gt_paths]

    ED = max(volumes)
    ES = min(volumes)

    ef = (ED - ES) / (ED + 1e-8)

    return ef


# =========================
# MAIN
# =========================
def main():
    data_dir = os.path.expanduser(
        "~/Desktop/aurora-github/aurora_v2_validation/data"
    )

    output_csv = os.path.expanduser(
        "~/Desktop/aurora-github/aurora_v2_validation/outputs/ef_results.csv"
    )

    results = []

    for patient in sorted(os.listdir(data_dir)):
        p_path = os.path.join(data_dir, patient)

        if not os.path.isdir(p_path):
            continue

        ef = compute_ef(p_path)

        if ef is not None:
            print(f"{patient} → EF: {ef:.3f}")
            results.append([patient, ef])

    # save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Patient", "EF"])
        writer.writerows(results)

    print(f"\nSaved EF results: {output_csv}")


if __name__ == "__main__":
    main()
