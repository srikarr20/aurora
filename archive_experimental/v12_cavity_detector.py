import numpy as np
import pandas as pd
import nibabel as nib
import os
import cv2

# =========================
# LOAD MRI
# =========================
def load_4d(path):
    data = np.squeeze(nib.load(path).get_fdata())

    if data.ndim != 4:
        raise ValueError(f"Expected (H,W,Z,T), got {data.shape}")

    return data


# =========================
# SIMPLE CAVITY DETECTION
# =========================
def detect_cavity(frame):

    # normalize
    img = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # invert (blood pool is dark)
    img_inv = 255 - img

    # threshold (adaptive)
    _, mask = cv2.threshold(img_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return None

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cavity = (labels == largest).astype(np.uint8)

    return cavity


# =========================
# VOLUME PROXY PER SLICE
# =========================
def compute_slice_volume(slice_data):

    T = slice_data.shape[2]
    volumes = []

    for t in range(T):

        frame = slice_data[:, :, t]
        cavity = detect_cavity(frame)

        if cavity is None:
            continue

        volumes.append(np.sum(cavity))

    if len(volumes) < 2:
        return None

    return np.array(volumes)


# =========================
# EXTRACT FEATURES
# =========================
def extract_features(path):

    data = load_4d(path)
    H, W, Z, T = data.shape

    all_volumes = []

    for z in range(Z):

        slice_data = data[:, :, z, :]
        vol = compute_slice_volume(slice_data)

        if vol is not None:
            all_volumes.append(vol)

    if len(all_volumes) == 0:
        return None

    # align lengths (truncate to shortest)
    min_len = min(len(v) for v in all_volumes)
    vols = np.array([v[:min_len] for v in all_volumes])

    # aggregate across slices
    volume_curve = np.mean(vols, axis=0)

    v_max = np.max(volume_curve)
    v_min = np.min(volume_curve)

    ef = (v_max - v_min) / (v_max + 1e-6)

    return {
        "V_max": v_max,
        "V_min": v_min,
        "EF_est": ef,
        "Curve_std": np.std(volume_curve)
    }


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
            feat = extract_features(nii_path)
            if feat is None:
                continue
        except Exception as e:
            print("Skipping:", patient, "|", e)
            continue

        row = {"Patient": patient}
        row.update(feat)

        rows.append(row)

        print("Processed:", patient)

    df = pd.DataFrame(rows)

    os.makedirs("outputs_v12", exist_ok=True)
    df.to_csv("outputs_v12/results.csv", index=False)

    print("\nSaved: outputs_v12/results.csv")
    print("Total processed:", len(df))


if __name__ == "__main__":
    process("data")
