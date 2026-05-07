import numpy as np
import pandas as pd
import nibabel as nib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =========================
# LOAD MRI
# =========================
def load_4d(path):
    data = np.squeeze(nib.load(path).get_fdata())
    return data

# =========================
# SIGNED DELTA (KEY CHANGE)
# =========================
def compute_signed_delta(data):
    return np.diff(data, axis=3)  # NO abs()

# =========================
# REGIONAL SIGNALS (MEAN)
# =========================
def regional_rho(delta, splits=4):

    H, W, Z, T = delta.shape
    h = H // splits
    w = W // splits

    signals = []

    for i in range(splits):
        for j in range(splits):
            patch = delta[i*h:(i+1)*h, j*w:(j+1)*w, :, :]
            rho_patch = np.mean(patch, axis=(0,1,2))
            signals.append(rho_patch)

    return np.array(signals)

# =========================
# FEATURE EXTRACTION
# =========================
def compute_features(regional):

    eps = 1e-6

    # ---- GLOBAL SIGNAL
    global_signal = np.mean(regional, axis=0)

    # ---- BASIC FEATURES (V17.3)
    deviation = np.std(global_signal) / (np.mean(np.abs(global_signal)) + eps)

    spatial_var = np.var(regional, axis=0).mean()
    spatial_var /= (np.mean(np.abs(global_signal))**2 + eps)

    temporal_instability = np.std(np.diff(global_signal))
    temporal_instability /= (np.mean(np.abs(global_signal)) + eps)

    # =========================
    # NEW FEATURES (V18)
    # =========================

    # ---- Phase consistency (periodicity)
    fft = np.fft.fft(global_signal)
    power = np.abs(fft)

    dominant = np.max(power[1:])  # ignore DC
    total = np.sum(power[1:]) + eps

    phase_consistency = dominant / total

    # ---- Contraction bias (signed motion)
    contraction_bias = np.mean(global_signal) / (np.mean(np.abs(global_signal)) + eps)

    return deviation, spatial_var, temporal_instability, phase_consistency, contraction_bias

# =========================
# PROCESS PATIENT
# =========================
def process_patient(path):

    data = load_4d(path)
    delta = compute_signed_delta(data)

    regional = regional_rho(delta)

    return compute_features(regional)

# =========================
# DATASET LOOP
# =========================
def process_dataset(data_dir):

    results = []

    for patient in sorted(os.listdir(data_dir)):

        p_dir = os.path.join(data_dir, patient)
        if not os.path.isdir(p_dir):
            continue

        nii_path = os.path.join(p_dir, f"{patient}_4d.nii.gz")
        if not os.path.exists(nii_path):
            continue

        try:
            d, s, t, p, c = process_patient(nii_path)
        except Exception as e:
            print("Skipping:", patient, e)
            continue

        results.append({
            "Patient": patient,
            "Deviation": d,
            "SpatialVar": s,
            "TemporalInstability": t,
            "PhaseConsistency": p,
            "ContractionBias": c
        })

        print("Processed:", patient)

    df = pd.DataFrame(results)

    os.makedirs("outputs_v18", exist_ok=True)
    df.to_csv("outputs_v18/features.csv", index=False)

    print("\nSaved: outputs_v18/features.csv")

    return df

# =========================
# ANALYSIS
# =========================
def analyze(df):

    X = df[[
        "Deviation",
        "SpatialVar",
        "TemporalInstability",
        "PhaseConsistency",
        "ContractionBias"
    ]].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=0)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    print("\n=== CLUSTER COUNTS ===")
    print(df["Cluster"].value_counts())

    # 2D visualization
    plt.figure()

    colors = ["green", "orange", "red"]

    for c in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == c]
        plt.scatter(
            subset["Deviation"],
            subset["PhaseConsistency"],
            color=colors[c],
            label=f"Cluster {c}"
        )

    plt.xlabel("Deviation")
    plt.ylabel("PhaseConsistency")
    plt.title("V18 Phase-Aware State Space")
    plt.legend()
    plt.savefig("outputs_v18/state_space.png")

    df.to_csv("outputs_v18/results.csv", index=False)

    print("\nSaved: outputs_v18/results.csv")

# =========================
# RUN
# =========================
if __name__ == "__main__":

    df = process_dataset("data")
    analyze(df)
