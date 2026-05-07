import numpy as np
import pandas as pd
import nibabel as nib
import os
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

# =========================
# LOAD MRI
# =========================
def load_4d(path):
    data = np.squeeze(nib.load(path).get_fdata())
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got {data.shape}")
    return data


# =========================
# DETECTOR (ΔV)
# =========================
def compute_deltaV(data):
    # data: (H,W,Z,T)
    return np.abs(np.diff(data, axis=3))


# =========================
# GLOBAL SIGNAL ρ(t)
# =========================
def compute_rho(deltaV):
    return np.sum(deltaV, axis=(0,1,2))


# =========================
# OBSERVABLES
# =========================
def compute_observables(deltaV, rho):

    T = len(rho)

    C = []
    K = []
    E = []

    for t in range(T):

        frame = deltaV[:,:,:,t]

        # coherence: spatial uniformity
        C.append(np.std(frame) / (np.mean(frame) + 1e-6))

        # variability: frame-to-frame change
        if t > 0:
            K.append(np.std(frame - deltaV[:,:,:,t-1]))
        else:
            K.append(0)

        # energy: total activity
        E.append(np.sum(frame))

    return np.array(C), np.array(K), np.array(E)


# =========================
# STATE SPACE
# =========================
def build_state(C, K, E):

    # normalize
    def norm(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-6)

    Cn = norm(C)
    Kn = norm(K)
    En = norm(E)

    X = np.vstack([Cn, Kn, En]).T
    return X


# =========================
# CYCLE DETECTION
# =========================
def detect_cycles(rho):

    peaks, _ = find_peaks(rho, distance=3)

    cycles = []

    for i in range(len(peaks)-1):
        cycles.append((peaks[i], peaks[i+1]))

    return cycles


# =========================
# ATTRACTOR METRICS
# =========================
def attractor_metrics(X):

    # compactness
    centroid = np.mean(X, axis=0)
    dist = np.linalg.norm(X - centroid, axis=1)
    compactness = np.mean(dist)

    # smoothness (temporal continuity)
    velocity = np.diff(X, axis=0)
    smoothness = np.mean(np.linalg.norm(velocity, axis=1))

    return compactness, smoothness


# =========================
# HEALTH REFERENCE
# =========================
def build_health_reference(df_states):

    # average trajectory (simple version)
    all_states = np.vstack(df_states)
    mean = np.mean(all_states, axis=0)

    return mean


# =========================
# DEVIATION
# =========================
def compute_deviation(X, X_ref):

    d = np.linalg.norm(X - X_ref, axis=1)

    mean_d = np.mean(d)

    # trend
    trend = np.mean(np.diff(d))

    return d, mean_d, trend


# =========================
# DECISION LAYER
# =========================
def classify(mean_d, trend):

    if mean_d < 0.5 and abs(trend) < 0.05:
        return "STABLE"
    elif mean_d < 1.0:
        return "UNSTABLE"
    else:
        return "DYSFUNCTIONAL"


# =========================
# PROCESS PATIENT
# =========================
def process_patient(path):

    data = load_4d(path)

    deltaV = compute_deltaV(data)
    rho = compute_rho(deltaV)

    C, K, E = compute_observables(deltaV, rho)
    X = build_state(C, K, E)

    compactness, smoothness = attractor_metrics(X)

    return {
        "X": X,
        "rho": rho,
        "compactness": compactness,
        "smoothness": smoothness
    }


# =========================
# MAIN PIPELINE
# =========================
def process_dataset(data_dir):

    states = []
    rows = []

    for patient in sorted(os.listdir(data_dir)):

        p_dir = os.path.join(data_dir, patient)
        if not os.path.isdir(p_dir):
            continue

        nii_path = os.path.join(p_dir, f"{patient}_4d.nii.gz")

        if not os.path.exists(nii_path):
            continue

        try:
            result = process_patient(nii_path)
        except Exception as e:
            print("Skipping:", patient, e)
            continue

        states.append(result["X"])

        rows.append({
            "Patient": patient,
            "Compactness": result["compactness"],
            "Smoothness": result["smoothness"]
        })

        print("Processed:", patient)

    # build reference
    X_ref = build_health_reference(states)

    # second pass for deviation
    final_rows = []

    for i, patient in enumerate(sorted(os.listdir(data_dir))):

        if i >= len(states):
            continue

        X = states[i]

        d, mean_d, trend = compute_deviation(X, X_ref)

        label = classify(mean_d, trend)

        final_rows.append({
            "Patient": patient,
            "MeanDeviation": mean_d,
            "Trend": trend,
            "State": label
        })

    df = pd.DataFrame(final_rows)

    os.makedirs("outputs_final", exist_ok=True)
    df.to_csv("outputs_final/results.csv", index=False)

    print("\nSaved: outputs_final/results.csv")
    print("Total:", len(df))


# =========================
# RUN
# =========================
if __name__ == "__main__":
    process_dataset("data")
