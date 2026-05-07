import numpy as np
import pandas as pd
import nibabel as nib
import os
import cv2

# =========================
# LOAD MRI + ED MASK
# =========================
def load_data(patient_dir, patient):

    img_path = os.path.join(patient_dir, f"{patient}_4d.nii.gz")
    data = np.squeeze(nib.load(img_path).get_fdata())

    mask_path = os.path.join(patient_dir, f"{patient}_frame01_gt.nii.gz")

    if not os.path.exists(mask_path):
        return None, None

    mask = np.squeeze(nib.load(mask_path).get_fdata())

    return data, mask


# =========================
# FORCES (TRUE OPERATORS)
# =========================

def intensity_force(frame):
    f = cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX)
    return (1 - f)


def curvature_force(region):
    gx, gy = np.gradient(region)
    gxx, _ = np.gradient(gx)
    _, gyy = np.gradient(gy)
    return gxx + gyy  # Laplacian


def temporal_force(region, prev):
    return region - prev


# =========================
# PDE EVOLUTION
# =========================
def evolve_pde(slice_data, init_mask):

    H, W, T = slice_data.shape

    region = init_mask.astype(np.float32)
    region = region / (np.max(region) + 1e-6)

    prev_region = region.copy()

    volumes = []

    # PDE coefficients (important)
    alpha = 0.6   # intensity
    beta  = 0.25  # curvature
    gamma = 0.15  # temporal

    for t in range(T):

        frame = slice_data[:,:,t]

        I = intensity_force(frame)
        C = curvature_force(region)

        if t > 0:
            T_force = temporal_force(region, prev_region)
        else:
            T_force = 0

        # TRUE EVOLUTION STEP
        update = (
            alpha * I +
            beta  * C -
            gamma * T_force
        )

        prev_region = region.copy()

        region = region + update

        # stability clamp (not smoothing)
        region = np.clip(region, 0, None)
        region = region / (np.max(region) + 1e-6)

        # threshold (adaptive, no blur)
        th = np.percentile(region, 85)
        mask = region > th

        vol = np.sum(mask)
        volumes.append(vol)

    return np.array(volumes)


# =========================
# PROCESS PATIENT
# =========================
def process_patient(patient_dir, patient):

    data, mask = load_data(patient_dir, patient)

    if data is None:
        return None

    H, W, Z, T = data.shape

    curves = []

    for z in range(Z):

        if z >= mask.shape[2]:
            continue

        init_mask = mask[:,:,z] > 0

        if np.sum(init_mask) < 20:
            continue

        slice_data = data[:,:,z,:]

        vol = evolve_pde(slice_data, init_mask)

        if len(vol) > 5:
            curves.append(vol)

    if len(curves) == 0:
        return None

    min_len = min(len(c) for c in curves)
    curves = np.array([c[:min_len] for c in curves])

    volume_curve = np.mean(curves, axis=0)

    # EF (now valid — identity is fixed)
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

        patient_dir = os.path.join(data_dir, patient)

        if not os.path.isdir(patient_dir):
            continue

        try:
            feat = process_patient(patient_dir, patient)
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

    os.makedirs("outputs_v16_1", exist_ok=True)
    df.to_csv("outputs_v16_1/results.csv", index=False)

    print("\nSaved: outputs_v16_1/results.csv")
    print("Total processed:", len(df))


if __name__ == "__main__":
    process("data")
