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
# OPTICAL FLOW (Farneback)
# =========================
def compute_flow(frame1, frame2):

    f1 = cv2.normalize(frame1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    f2 = cv2.normalize(frame2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    flow = cv2.calcOpticalFlowFarneback(
        f1, f2, None,
        pyr_scale=0.5,
        levels=2,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    return flow  # (H,W,2)


# =========================
# DIVERGENCE (volume proxy)
# =========================
def compute_divergence(flow):

    fx = flow[...,0]
    fy = flow[...,1]

    dfx_dx = np.gradient(fx, axis=1)
    dfy_dy = np.gradient(fy, axis=0)

    return dfx_dx + dfy_dy


# =========================
# PROCESS ONE PATIENT
# =========================
def extract_features(path):

    data = load_4d(path)
    H, W, Z, T = data.shape

    contractions = []
    expansions = []
    coherences = []

    for z in range(Z):

        slice_data = data[:,:,z,:]

        mean_img = np.mean(slice_data, axis=2)
        thresh = np.percentile(mean_img, 60)
        mask = (mean_img > thresh)

        if np.sum(mask) < 50:
            continue

        for t in range(T-1):

            f1 = slice_data[:,:,t]
            f2 = slice_data[:,:,t+1]

            flow = compute_flow(f1, f2)

            div = compute_divergence(flow)

            # contraction = negative divergence
            contractions.append(np.mean(div[mask & (div < 0)]))

            # expansion = positive divergence
            expansions.append(np.mean(div[mask & (div > 0)]))

            # coherence = spatial consistency
            mag = np.linalg.norm(flow, axis=2)
            coherences.append(np.std(mag[mask]))

    if len(contractions) == 0:
        return None

    return {
        "CON_mean": np.nanmean(contractions),
        "EXP_mean": np.nanmean(expansions),
        "COH_mean": np.nanmean(coherences),
        "CON_std": np.nanstd(contractions),
        "EXP_std": np.nanstd(expansions),
        "COH_std": np.nanstd(coherences),
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

    os.makedirs("outputs_v11", exist_ok=True)
    df.to_csv("outputs_v11/features.csv", index=False)

    print("\nSaved: outputs_v11/features.csv")
    print("Total processed:", len(df))


if __name__ == "__main__":
    process("data")
