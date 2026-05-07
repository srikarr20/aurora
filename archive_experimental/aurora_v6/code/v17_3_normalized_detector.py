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
    if data.ndim != 4:
        raise ValueError(f"Expected 4D, got {data.shape}")
    return data

# =========================
# DELTA V
# =========================
def compute_deltaV(data):
    return np.abs(np.diff(data, axis=3))

# =========================
# REGIONAL SIGNALS (NORMALIZED)
# =========================
def regional_rho(deltaV, splits=4):
    H, W, Z, T = deltaV.shape
    h = H // splits
    w = W // splits

    signals = []

    for i in range(splits):
        for j in range(splits):
            patch = deltaV[i*h:(i+1)*h, j*w:(j+1)*w, :, :]
            
            # ✅ FIX 1: mean instead of sum (scale-invariant)
            rho_patch = np.mean(patch, axis=(0,1,2))
            
            signals.append(rho_patch)

    return np.array(signals)  # (regions, T)

# =========================
# FEATURES (NORMALIZED)
# =========================
def compute_features(regional_signals):

    # ✅ FIX 2: normalized global signal
    global_signal = np.mean(regional_signals, axis=0)

    eps = 1e-6

    # ---- Deviation (already normalized)
    deviation = np.std(global_signal) / (np.mean(global_signal) + eps)

    # ---- Spatial variance (dimensionless)
    spatial_var = np.var(regional_signals, axis=0).mean()
    spatial_var = spatial_var / ((np.mean(global_signal) ** 2) + eps)

    # ---- Temporal instability (dimensionless)
    temporal_instability = np.std(np.diff(global_signal))
    temporal_instability = temporal_instability / (np.mean(global_signal) + eps)

    return deviation, spatial_var, temporal_instability

# =========================
# PROCESS PATIENT
# =========================
def process_patient(path):

    data = load_4d(path)
    deltaV = compute_deltaV(data)

    regional = regional_rho(deltaV, splits=4)

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
            d, s, t = process_patient(nii_path)
        except Exception as e:
            print("Skipping:", patient, e)
            continue

        results.append({
            "Patient": patient,
            "Deviation": d,
            "SpatialVar": s,
            "TemporalInstability": t
        })

        print("Processed:", patient)

    df = pd.DataFrame(results)

    os.makedirs("outputs_v17_3", exist_ok=True)
    df.to_csv("outputs_v17_3/features.csv", index=False)

    print("\nSaved: outputs_v17_3/features.csv")

    return df

# =========================
# ANALYSIS + CLUSTERING
# =========================
def analyze(df):

    X = df[["Deviation", "SpatialVar", "TemporalInstability"]].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=0)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    print("\n=== CLUSTER COUNTS ===")
    print(df["Cluster"].value_counts())

    # 2D plot
    plt.figure()

    colors = ["green", "orange", "red"]

    for c in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == c]
        plt.scatter(
            subset["Deviation"],
            subset["SpatialVar"],
            color=colors[c],
            label=f"Cluster {c}"
        )

    plt.xlabel("Deviation")
    plt.ylabel("SpatialVar")
    plt.title("V17.3 Normalized State Space")
    plt.legend()
    plt.savefig("outputs_v17_3/state_space.png")

    # 3D plot
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for c in sorted(df["Cluster"].unique()):
            subset = df[df["Cluster"] == c]
            ax.scatter(
                subset["Deviation"],
                subset["SpatialVar"],
                subset["TemporalInstability"],
                color=colors[c],
                label=f"Cluster {c}"
            )

        ax.set_xlabel("Deviation")
        ax.set_ylabel("SpatialVar")
        ax.set_zlabel("TemporalInstability")
        plt.title("3D Normalized State Space")
        plt.legend()
        plt.savefig("outputs_v17_3/state_space_3d.png")
    except:
        pass

    df.to_csv("outputs_v17_3/results.csv", index=False)

    print("\nSaved: outputs_v17_3/results.csv")

# =========================
# RUN
# =========================
if __name__ == "__main__":

    df = process_dataset("data")
    analyze(df)
