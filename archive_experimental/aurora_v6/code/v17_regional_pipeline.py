import numpy as np
import pandas as pd
import nibabel as nib
import os

# =========================
# LOAD MRI
# =========================
def load_4d(path):
    data = np.squeeze(nib.load(path).get_fdata())
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got {data.shape}")
    return data


# =========================
# DETECTOR
# =========================
def compute_deltaV(data):
    return np.abs(np.diff(data, axis=3))


# =========================
# REGIONAL DETECTOR
# =========================
def regional_rho(deltaV, splits=4):
    H, W, Z, T = deltaV.shape

    h = H // splits
    w = W // splits

    signals = []

    for i in range(splits):
        for j in range(splits):
            patch = deltaV[
                i*h:(i+1)*h,
                j*w:(j+1)*w,
                :, :
            ]

            rho_patch = np.sum(patch, axis=(0,1,2))
            signals.append(rho_patch)

    return np.array(signals)


# =========================
# OBSERVABLES
# =========================
def compute_regional_observables(regional_signals):

    num_regions, T = regional_signals.shape

    C_all, K_all, E_all = [], [], []

    for r in range(num_regions):

        rho = regional_signals[r]

        C = np.std(rho) / (np.mean(rho) + 1e-6)
        K = np.std(np.diff(rho))
        E = np.sum(rho)

        C_all.append(C)
        K_all.append(K)
        E_all.append(E)

    return np.array(C_all), np.array(K_all), np.array(E_all)


# =========================
# STATE VECTOR
# =========================
def build_state(C, K, E):

    def norm(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-6)

    Cn = norm(C)
    Kn = norm(K)
    En = norm(E)

    return np.concatenate([Cn, Kn, En])


# =========================
# REFERENCE
# =========================
def build_reference(states):
    return np.mean(np.vstack(states), axis=0)


# =========================
# DEVIATION
# =========================
def compute_deviation(X, X_ref):
    return np.linalg.norm(X - X_ref)


# =========================
# CLASSIFICATION
# =========================
def classify(dev, all_devs):

    q1 = np.percentile(all_devs, 33)
    q2 = np.percentile(all_devs, 66)

    if dev < q1:
        return "STABLE"
    elif dev < q2:
        return "UNSTABLE"
    else:
        return "DYSFUNCTIONAL"


# =========================
# PROCESS PATIENT
# =========================
def process_patient(path):

    data = load_4d(path)
    deltaV = compute_deltaV(data)

    regional = regional_rho(deltaV, splits=4)
    C, K, E = compute_regional_observables(regional)

    return build_state(C, K, E)


# =========================
# MAIN
# =========================
def process_dataset(data_dir):

    states, patients = [], []

    for patient in sorted(os.listdir(data_dir)):

        p_dir = os.path.join(data_dir, patient)
        if not os.path.isdir(p_dir):
            continue

        nii_path = os.path.join(p_dir, f"{patient}_4d.nii.gz")

        if not os.path.exists(nii_path):
            continue

        try:
            X = process_patient(nii_path)
        except Exception as e:
            print("Skipping:", patient, e)
            continue

        states.append(X)
        patients.append(patient)
        print("Processed:", patient)

    X_ref = build_reference(states)

    deviations = [compute_deviation(X, X_ref) for X in states]

    results = []

    for i, patient in enumerate(patients):

        dev = deviations[i]
        label = classify(dev, deviations)

        results.append({
            "Patient": patient,
            "Deviation": dev,
            "State": label
        })

    df = pd.DataFrame(results)

    os.makedirs("outputs_v17", exist_ok=True)
    df.to_csv("outputs_v17/results.csv", index=False)

    print("\nSaved: outputs_v17/results.csv")


if __name__ == "__main__":
    process_dataset("data")
