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
# OPTICAL FLOW
# =========================
def compute_flow(f1, f2):

    f1 = cv2.normalize(f1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    f2 = cv2.normalize(f2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        f1, f2, None,
        0.5, 2, 15, 3, 5, 1.2, 0
    )

    mag = np.linalg.norm(flow, axis=2)

    return mag


# =========================
# CENTRAL PRIOR
# =========================
def center_weight(H, W):

    y, x = np.ogrid[:H, :W]
    cy, cx = H//2, W//2

    dist = np.sqrt((x-cx)**2 + (y-cy)**2)
    dist = dist / np.max(dist)

    return 1 - dist  # center = high weight


# =========================
# DETECT CAVITY (KEY)
# =========================
def detect_cavity(slice_data):

    H, W, T = slice_data.shape

    # 1. intensity (mean)
    mean_img = np.mean(slice_data, axis=2)

    # normalize
    img = cv2.normalize(mean_img, None, 0, 255, cv2.NORM_MINMAX)

    # invert (blood = dark)
    inv = 255 - img

    # threshold
    _, intensity_mask = cv2.threshold(
        inv.astype(np.uint8),
        0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    intensity_mask = intensity_mask.astype(bool)

    # 2. motion map
    motion = np.zeros((H, W))

    for t in range(T-1):
        motion += compute_flow(slice_data[:,:,t], slice_data[:,:,t+1])

    motion = motion / (T-1)

    # normalize motion
    motion = motion / (np.max(motion) + 1e-6)

    motion_mask = motion > np.percentile(motion, 70)

    # 3. center prior
    center = center_weight(H, W)
    center_mask = center > 0.5

    # 4. combine
    combined = intensity_mask & motion_mask & center_mask

    if np.sum(combined) < 50:
        return None

    # 5. keep largest component
    labels = cv2.connectedComponents(combined.astype(np.uint8))[1]

    unique, counts = np.unique(labels, return_counts=True)

    if len(unique) <= 1:
        return None

    largest = unique[1:][np.argmax(counts[1:])]
    cavity = (labels == largest)

    return cavity


# =========================
# VOLUME PER SLICE
# =========================
def compute_slice_volume(slice_data):

    cavity = detect_cavity(slice_data)

    if cavity is None:
        return None

    T = slice_data.shape[2]
    volumes = []

    for t in range(T):

        frame = slice_data[:,:,t]
        masked = frame * cavity

        # dynamic threshold inside cavity
        thresh = np.percentile(masked[cavity], 50)

        region = (frame < thresh) & cavity
        volumes.append(np.sum(region))

    return np.array(volumes)


# =========================
# EXTRACT FEATURES
# =========================
def extract_features(path):

    data = load_4d(path)
    H, W, Z, T = data.shape

    all_volumes = []

    for z in range(Z):

        vol = compute_slice_volume(data[:,:,z,:])

        if vol is not None and len(vol) > 2:
            all_volumes.append(vol)

    if len(all_volumes) == 0:
        return None

    # align time
    min_len = min(len(v) for v in all_volumes)
    vols = np.array([v[:min_len] for v in all_volumes])

    volume_curve = np.mean(vols, axis=0)

    v_max = np.max(volume_curve)
    v_min = np.min(volume_curve)

    ef = (v_max - v_min) / (v_max + 1e-6)

    return {
        "EF_est": ef,
        "V_max": v_max,
        "V_min": v_min,
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

    os.makedirs("outputs_v12_1", exist_ok=True)
    df.to_csv("outputs_v12_1/results.csv", index=False)

    print("\nSaved: outputs_v12_1/results.csv")
    print("Total processed:", len(df))


if __name__ == "__main__":
    process("data")
