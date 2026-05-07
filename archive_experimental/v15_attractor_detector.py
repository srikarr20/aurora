import numpy as np
import pandas as pd
import nibabel as nib
import os
import cv2
from scipy.ndimage import gaussian_filter

# =========================
# LOAD MRI
# =========================
def load_4d(path):
    data = np.squeeze(nib.load(path).get_fdata())
    if data.ndim != 4:
        raise ValueError(f"Expected (H,W,Z,T), got {data.shape}")
    return data


# =========================
# INITIAL REGION (soft start)
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
    return (region - prev)


def cyclic_term(history):
    if len(history) < 5:
        return 0

    signal = np.array(history)
    fft = np.fft.fft(signal)
    power = np.abs(fft)
    power[0] = 0

    periodicity = np.max(power) / (np.sum(power) + 1e-6)

    return periodicity


# =========================
# REGION EVOLUTION
# =========================
def evolve_region(slice_data):

    H, W, T = slice_data.shape

    region = init_region(slice_data[:,:,0])

    history = []
    volumes = []

    # weights (tunable)
    lam1 = 2.0  # intensity
    lam2 = 1.0  # smoothness
    lam3 = 1.0  # temporal
    lam4 = 2.0  # cyclic

    for t in range(T):

        frame = slice_data[:,:,t]

        I_term = intensity_term(frame)
        S_term = smoothness_term(region)
        T_term = temporal_term(region, region)

        cyc = cyclic_term(history)

        # nonlinear update
        region = (
            lam1 * I_term +
            lam2 * S_term -
            lam3 * T_term +
            lam4 * cyc
        )

        # normalize
        region = gaussian_filter(region, sigma=1)
        region = region / (np.max(region) + 1e-6)

        # threshold
        mask = region > 0.5

        vol = np.sum(mask)

        history.append(vol)
        volumes.append(vol)

    return np.array(volumes)


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

    v_max = np.max(volume_curve)
    v_min = np.min(volume_curve)

    ef = (v_max - v_min) / (v_max + 1e-6)

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

    os.makedirs("outputs_v15", exist_ok=True)
    df.to_csv("outputs_v15/results.csv", index=False)

    print("\nSaved: outputs_v15/results.csv")
    print("Total processed:", len(df))


if __name__ == "__main__":
    process("data")
