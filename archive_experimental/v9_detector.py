import numpy as np
import pandas as pd
import nibabel as nib
import os

# =========================
# LOAD MRI (NO COLLAPSE)
# =========================
def load_4d(path):

    nii = nib.load(path)
    data = nii.get_fdata()
    data = np.squeeze(data)

    if data.ndim != 4:
        raise ValueError(f"Expected 4D (H,W,Z,T), got {data.shape}")

    return data  # (H, W, Z, T)


# =========================
# SPLIT PATCHES (2D)
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
# ACCUMULATION (PER SLICE)
# =========================
def compute_accumulation_slice(data_slice):

    acc = np.zeros(data_slice.shape[:2])

    for t in range(data_slice.shape[2] - 1):
        acc += np.abs(data_slice[:, :, t+1] - data_slice[:, :, t])

    return acc


# =========================
# EXTRACT FEATURES (V9)
# =========================
def extract_features(path):

    data = load_4d(path)

    H, W, Z, T = data.shape

    features = []

    for z in range(Z):

        slice_data = data[:, :, z, :]  # (H,W,T)

        # mean image for mask
        mean_img = np.mean(slice_data, axis=2)

        thresh = np.percentile(mean_img, 60)
        mask = (mean_img > thresh).astype(np.float32)

        if np.sum(mask) < 50:
            continue  # skip weak slice

        patches = split_patches(mask, 2)
        acc = compute_accumulation_slice(slice_data)

        for r, p in enumerate(patches):

            if np.sum(p) < 10:
                continue

            # ACI
            region = acc[p > 0]
            mean = np.mean(region)
            std  = np.std(region)

            aci = 0 if std == 0 else mean / (std + 1e-6)

            # TCI
            signal = [
                np.mean(slice_data[:, :, t][p > 0])
                for t in range(T)
            ]

            signal = np.array(signal)

            tci = 0 if len(signal) < 2 else np.std(np.diff(signal))

            features.append({
                "z": z,
                "region": r,
                "ACI": aci,
                "TCI": tci
            })

    return features


# =========================
# AGGREGATE TO FIXED VECTOR
# =========================
def aggregate_features(feature_list):

    if len(feature_list) == 0:
        return None

    df = pd.DataFrame(feature_list)

    # aggregate statistics
    agg = {}

    agg["ACI_mean"] = df["ACI"].mean()
    agg["ACI_std"]  = df["ACI"].std()
    agg["ACI_max"]  = df["ACI"].max()

    agg["TCI_mean"] = df["TCI"].mean()
    agg["TCI_std"]  = df["TCI"].std()
    agg["TCI_max"]  = df["TCI"].max()

    # heterogeneity (important!)
    agg["ACI_range"] = df["ACI"].max() - df["ACI"].min()
    agg["TCI_range"] = df["TCI"].max() - df["TCI"].min()

    return agg


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
            features = extract_features(nii_path)
            agg = aggregate_features(features)

            if agg is None:
                continue

        except Exception as e:
            print("Skipping:", patient, "|", e)
            continue

        row = {"Patient": patient}
        row.update(agg)

        rows.append(row)

        print("Processed:", patient)

    df = pd.DataFrame(rows)

    os.makedirs("outputs_v9", exist_ok=True)
    save_path = "outputs_v9/features.csv"

    df.to_csv(save_path, index=False)

    print("\nSaved:", save_path)
    print("Total processed:", len(df))


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    process("data")
