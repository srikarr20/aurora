import numpy as np
import pandas as pd
import nibabel as nib
import os

# =========================
# LOAD MRI
# =========================
def load_4d(path):
    nii = nib.load(path)
    data = np.squeeze(nii.get_fdata())

    if data.ndim != 4:
        raise ValueError(f"Expected 4D (H,W,Z,T), got {data.shape}")

    return data


# =========================
# GLOBAL SIGNAL PER SLICE
# =========================
def compute_slice_signal(slice_data, mask):

    signal = []

    for t in range(slice_data.shape[2]):
        frame = slice_data[:, :, t]

        if np.sum(mask) < 10:
            signal.append(0)
        else:
            signal.append(np.mean(frame[mask > 0]))

    return np.array(signal)


# =========================
# PERIODICITY (FFT)
# =========================
def compute_periodicity(signal):

    if len(signal) < 4:
        return 0

    fft = np.fft.fft(signal)
    power = np.abs(fft)

    # ignore DC component
    power[0] = 0

    return np.max(power) / (np.sum(power) + 1e-6)


# =========================
# PHASE STABILITY
# =========================
def compute_phase_stability(signal):

    if len(signal) < 3:
        return 0

    diff = np.diff(signal)
    return np.std(diff)


# =========================
# EXPANSION PROXY
# =========================
def compute_expansion(signal):

    if len(signal) < 2:
        return 0

    return np.max(signal) - np.min(signal)


# =========================
# PROCESS ONE PATIENT
# =========================
def extract_features(path):

    data = load_4d(path)
    H, W, Z, T = data.shape

    expansions = []
    periodicities = []
    stabilities = []

    for z in range(Z):

        slice_data = data[:, :, z, :]

        mean_img = np.mean(slice_data, axis=2)
        thresh = np.percentile(mean_img, 60)
        mask = (mean_img > thresh).astype(np.float32)

        if np.sum(mask) < 50:
            continue

        signal = compute_slice_signal(slice_data, mask)

        expansions.append(compute_expansion(signal))
        periodicities.append(compute_periodicity(signal))
        stabilities.append(compute_phase_stability(signal))

    if len(expansions) == 0:
        return None

    return {
        "EXP_mean": np.mean(expansions),
        "EXP_std": np.std(expansions),
        "PER_mean": np.mean(periodicities),
        "PER_std": np.std(periodicities),
        "STAB_mean": np.mean(stabilities),
        "STAB_std": np.std(stabilities),
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

    os.makedirs("outputs_v10", exist_ok=True)
    df.to_csv("outputs_v10/features.csv", index=False)

    print("\nSaved: outputs_v10/features.csv")
    print("Total processed:", len(df))


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    process("data")
