import numpy as np
import pandas as pd
import nibabel as nib
import os

# =========================
# LOAD MRI (FIXED)
# =========================
def load_4d(path):

    nii = nib.load(path)
    data = nii.get_fdata()

    data = np.squeeze(data)

    # =========================
    # CRITICAL FIX: HANDLE 4D (H,W,Z,T)
    # =========================
    if data.ndim == 4:
        # collapse Z dimension → (H,W,T)
        data = np.mean(data, axis=2)

    if data.ndim != 3:
        raise ValueError(f"Unsupported shape after processing: {data.shape}")

    return data


# =========================
# SPLIT PATCHES
# =========================
def split_patches(mask, n=2):

    H, W = mask.shape
    patches = []

    h_step = H // n
    w_step = W // n

    for i in range(n):
        for j in range(n):

            patch = np.zeros_like(mask)

            patch[
                i*h_step:(i+1)*h_step,
                j*w_step:(j+1)*w_step
            ] = mask[
                i*h_step:(i+1)*h_step,
                j*w_step:(j+1)*w_step
            ]

            patches.append(patch)

    return patches


# =========================
# ACCUMULATION
# =========================
def compute_accumulation(data):

    acc = np.zeros(data.shape[:2])

    for t in range(data.shape[2] - 1):
        acc += np.abs(data[:, :, t+1] - data[:, :, t])

    return acc


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(path):

    data = load_4d(path)

    H, W, T = data.shape

    # robust projection image
    mean_img = np.mean(data, axis=2)

    # robust mask
    thresh = np.percentile(mean_img, 60)
    mask = (mean_img > thresh).astype(np.float32)

    if np.sum(mask) < 50:
        raise ValueError("Mask too small")

    patches = split_patches(mask, 2)
    acc = compute_accumulation(data)

    aci, tci = [], []

    for p in patches:

        if np.sum(p) < 10:
            aci.append(0)
            tci.append(0)
            continue

        region = acc[p > 0]

        mean = np.mean(region)
        std  = np.std(region)

        if std == 0 or np.isnan(std):
            aci.append(0)
        else:
            aci.append(mean / (std + 1e-6))

        signal = [
            np.mean(data[:, :, t][p > 0])
            for t in range(T)
        ]

        signal = np.array(signal)

        if len(signal) < 2:
            tci.append(0)
        else:
            tci.append(np.std(np.diff(signal)))

    return aci, tci


# =========================
# PROCESS DATASET
# =========================
def process(data_dir):

    rows = []

    for patient in sorted(os.listdir(data_dir)):

        p_dir = os.path.join(data_dir, patient)

        if not os.path.isdir(p_dir):
            continue

        nii_path = os.path.join(p_dir, f"{patient}_4d.nii.gz")

        if not os.path.exists(nii_path):
            continue

        try:
            aci, tci = extract_features(nii_path)
        except Exception as e:
            print("Skipping:", patient, "|", e)
            continue

        row = {"Patient": patient}

        for i in range(4):
            row[f"ACI_{i}"] = aci[i]
            row[f"TCI_{i}"] = tci[i]

        rows.append(row)

        print("Processed:", patient)

    df = pd.DataFrame(rows)

    os.makedirs("outputs_v8", exist_ok=True)
    save_path = "outputs_v8/aci_tci_regional.csv"

    df.to_csv(save_path, index=False)

    print("\nSaved:", save_path)
    print("Total processed:", len(df))


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    process("data")
