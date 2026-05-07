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
# INITIAL CAVITY (ED)
# =========================
def initial_cavity(frame):

    img = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inv = 255 - img

    _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return None

    H, W = frame.shape
    cy, cx = H//2, W//2

    best_score = -1
    best = None

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]
        y = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]//2
        x = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]//2

        dist = np.sqrt((x-cx)**2 + (y-cy)**2)

        score = area - 0.5 * dist

        if score > best_score:
            best_score = score
            best = (labels == i)

    return best.astype(np.uint8)


# =========================
# OPTICAL FLOW
# =========================
def flow(prev, curr):

    p = cv2.normalize(prev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    c = cv2.normalize(curr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    f = cv2.calcOpticalFlowFarneback(p, c, None,
                                    0.5, 2, 15, 3, 5, 1.2, 0)
    return f


# =========================
# WARP MASK
# =========================
def warp_mask(mask, flow):

    H, W = mask.shape

    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))

    map_x = (grid_x + flow[...,0]).astype(np.float32)
    map_y = (grid_y + flow[...,1]).astype(np.float32)

    warped = cv2.remap(mask.astype(np.float32),
                       map_x, map_y,
                       interpolation=cv2.INTER_NEAREST)

    return (warped > 0.5).astype(np.uint8)


# =========================
# REFINE MASK
# =========================
def refine_mask(frame, mask):

    vals = frame[mask > 0]

    if len(vals) < 10:
        return mask

    thresh = np.percentile(vals, 50)

    refined = (frame < thresh).astype(np.uint8)

    return refined * mask


# =========================
# TRACK ONE SLICE
# =========================
def track_slice(slice_data):

    H, W, T = slice_data.shape

    mask = initial_cavity(slice_data[:,:,0])
    if mask is None:
        return None

    volumes = []

    for t in range(T):

        if t > 0:
            f = flow(slice_data[:,:,t-1], slice_data[:,:,t])
            mask = warp_mask(mask, f)
            mask = refine_mask(slice_data[:,:,t], mask)

        area = np.sum(mask)

        # reject instability
        if len(volumes) > 0:
            prev = volumes[-1]
            if area > 2*prev or area < 0.5*prev:
                area = prev

        volumes.append(area)

    return np.array(volumes)


# =========================
# EXTRACT FEATURES
# =========================
def extract(path):

    data = load_4d(path)
    H, W, Z, T = data.shape

    curves = []

    for z in range(Z):

        vol = track_slice(data[:,:,z,:])

        if vol is not None and len(vol) > 3:
            curves.append(vol)

    if len(curves) == 0:
        return None

    min_len = min(len(c) for c in curves)
    curves = np.array([c[:min_len] for c in curves])

    curve = np.mean(curves, axis=0)

    v_max = np.max(curve)
    v_min = np.min(curve)

    ef = (v_max - v_min) / (v_max + 1e-6)

    return {
        "EF_est": ef,
        "V_max": v_max,
        "V_min": v_min,
        "Curve_std": np.std(curve)
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
            feat = extract(nii_path)
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

    os.makedirs("outputs_v12_2", exist_ok=True)
    df.to_csv("outputs_v12_2/results.csv", index=False)

    print("\nSaved: outputs_v12_2/results.csv")
    print("Total processed:", len(df))


if __name__ == "__main__":
    process("data")
