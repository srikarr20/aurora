import numpy as np
import pandas as pd
import nibabel as nib
import os
import cv2
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks

# =========================
# LOAD MRI
# =========================
def load_4d(path):
    data = np.squeeze(nib.load(path).get_fdata())
    if data.ndim != 4:
        raise ValueError(f"Expected (H,W,Z,T), got {data.shape}")
    return data


# =========================
# INITIAL REGION
# =========================
def init_region(frame):
    f = cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX)
    inv = 1 - f
    thresh = np.percentile(inv, 70)
    region = (inv > thresh).astype(np.float32)
    return gaussian_filter(region, sigma=2)


# =========================
# ENERGY TERMS
# =========================
def intensity_term(frame):
    f = cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX)
    return 1 - f


def smoothness_term(region):
    return gaussian_filter(region, sigma=1)


def temporal_term(region, prev):
    return region - prev


# =========================
# REGION EVOLUTION (V15 CORE)
# =========================
def evolve_region(slice_data):

    H, W, T = slice_data.shape

    region = init_region(slice_data[:,:,0])

    history_regions = []
    volume_history = []

    lam1 = 2.0
    lam2 = 1.0
    lam3 = 1.0

    for t in range(T):

        frame = slice_data[:,:,t]

        I_term = intensity_term(frame)
        S_term = smoothness_term(region)

        if len(history_regions) > 0:
            T_term = temporal_term(region, history_regions[-1])
        else:
            T_term = 0

        region = (
            lam1 * I_term +
            lam2 * S_term -
            lam3 * T_term
        )

        region = gaussian_filter(region, sigma=1)
        region = region / (np.max(region) + 1e-6)

        # tighter region (important for LV isolation)
        th = np.percentile(region, 80)
        mask = region > th

        vol = np.sum(mask)

        history_regions.append(region.copy())
        volume_history.append(vol)

    return np.array(volume_history)


# =========================
# V15.2 CORRECT EF EXTRACTION
# =========================
def compute_ef_stable(volume_curve):

    # -------------------------
    # 1. Light smoothing
    # -------------------------
    V = gaussian_filter1d(volume_curve, sigma=1)

    # -------------------------
    # 2. Find ED peaks
    # -------------------------
    peaks, _ = find_peaks(V, distance=3)

    # -------------------------
    # 3. Find ES troughs
    # -------------------------
    troughs, _ = find_peaks(-V, distance=3)

    # -------------------------
    # 4. Robust fallback
    # -------------------------
    if len(peaks) == 0 or len(troughs) == 0:
        v_max = np.percentile(V, 95)
        v_min = np.percentile(V, 5)
    else:
        v_max = np.median(V[peaks])
        v_min = np.median(V[troughs])

    # -------------------------
    # 5. EF
    # -------------------------
    ef = (v_max - v_min) / (v_max + 1e-6)

    return ef, v_max, v_min


# =========================
# PROCESS PATIENT
# =========================
def process_patient(path):

    data = load_4d(path)
    H, W, Z, T = data.shape

    curves = []

    for z in range(Z):

        vol = evolve_region(data[:,:,z,:])

        if len(vol) > 5:
            curves.append(vol)

    if len(curves) == 0:
        return None

    min_len = min(len(c) for c in curves)
    curves = np.array([c[:min_len] for c in curves])

    volume_curve = np.mean(curves, axis=0)

    ef, v_max, v_min = compute_ef_stable(volume_curve)

    return {
        "EF_est": ef,
        "V_max": v_max,
        "V_min": v_min
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
            feat = process_patient(nii_path)
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

    os.makedirs("outputs_v15_2", exist_ok=True)
    df.to_csv("outputs_v15_2/results.csv", index=False)

    print("\nSaved: outputs_v15_2/results.csv")
    print("Total processed:", len(df))


if __name__ == "__main__":
    process("data")
