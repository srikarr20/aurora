import numpy as np
import pandas as pd
import nibabel as nib
import os
import cv2
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# =========================
# LOAD MRI
# =========================
def load_4d(path):
    data = np.squeeze(nib.load(path).get_fdata())
    if data.ndim != 4:
        raise ValueError(f"Expected (H,W,Z,T), got {data.shape}")
    return data


# =========================
# CAVITY FIELD
# =========================
def compute_cavity_field(slice_data):

    H, W, T = slice_data.shape

    C = np.zeros((H, W, T))

    for t in range(T):
        frame = slice_data[:,:,t]

        # normalize
        f = cv2.normalize(frame, None, 0, 1, cv2.NORM_MINMAX)

        # invert (blood pool)
        inv = 1 - f

        # smooth
        inv = gaussian_filter(inv, sigma=1)

        C[:,:,t] = inv

    return C


# =========================
# MANIFOLD EXTRACTION
# =========================
def extract_manifold(C):

    H, W, T = C.shape

    # collapse along y-axis → 1D representation
    projection = np.mean(C, axis=0)  # shape (W, T)

    # transpose → (T, W)
    projection = projection.T

    # smooth
    projection = gaussian_filter(projection, sigma=1)

    # dominant trajectory
    x_star = np.argmax(projection, axis=1)

    # smooth trajectory
    x_star = gaussian_filter1d(x_star.astype(float), sigma=2)

    return x_star, projection


# =========================
# CURVATURE (κ)
# =========================
def compute_kappa(projection, x_star):

    grad = np.gradient(projection, axis=1)
    curvature = np.gradient(grad, axis=1)

    kappa = []

    T = projection.shape[0]

    for t in range(T):
        ix = int(np.clip(x_star[t], 0, projection.shape[1]-1))
        kappa.append(abs(curvature[t, ix]))

    kappa = np.array(kappa)
    kappa = kappa / (np.max(kappa) + 1e-6)
    kappa = gaussian_filter1d(kappa, sigma=3)

    return kappa


# =========================
# TRACKING (CRITICAL DAMPING)
# =========================
def track_manifold(x_star, kappa):

    T = len(x_star)

    x = x_star[0]
    v = 0.0

    traj = []

    alpha = 4.0
    beta = 1.2
    gamma = 0.75

    for t in range(T-1):

        drift = x_star[t+1] - x_star[t]

        force = alpha * drift - beta * kappa[t] * (x - x_star[t])

        v = (1 - gamma) * v + force
        x = x + v

        traj.append(x)

    traj = np.array(traj)

    return traj


# =========================
# WIDTH → VOLUME PROXY
# =========================
def compute_width(C, x_traj):

    H, W, T = C.shape

    widths = []

    for t in range(len(x_traj)):

        x = int(np.clip(x_traj[t], 0, W-1))

        column = C[:, x, t]

        thresh = np.percentile(column, 70)

        region = column > thresh

        width = np.sum(region)

        widths.append(width)

    return np.array(widths)


# =========================
# PROCESS PATIENT
# =========================
def process_patient(path):

    data = load_4d(path)
    H, W, Z, T = data.shape

    curves = []

    for z in range(Z):

        slice_data = data[:,:,z,:]

        C = compute_cavity_field(slice_data)

        x_star, proj = extract_manifold(C)

        kappa = compute_kappa(proj, x_star)

        x_traj = track_manifold(x_star, kappa)

        width = compute_width(C, x_traj)

        if len(width) > 5:
            curves.append(width)

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

    os.makedirs("outputs_v14", exist_ok=True)
    df.to_csv("outputs_v14/results.csv", index=False)

    print("\nSaved: outputs_v14/results.csv")
    print("Total processed:", len(df))


if __name__ == "__main__":
    process("data")
