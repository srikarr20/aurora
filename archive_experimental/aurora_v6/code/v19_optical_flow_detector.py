import numpy as np
import pandas as pd
import nibabel as nib
import os
import cv2
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
# OPTICAL FLOW (Farneback)
# =========================
def compute_flow_sequence(data):

    H, W, Z, T = data.shape

    flows = []

    for t in range(T - 1):

        frame1 = data[:, :, Z//2, t].astype(np.float32)
        frame2 = data[:, :, Z//2, t+1].astype(np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        flows.append(flow)

    return np.array(flows)  # (T-1, H, W, 2)

# =========================
# REGIONAL FEATURES
# =========================
def regional_flow_features(flows, splits=4):

    T, H, W, _ = flows.shape

    h = H // splits
    w = W // splits

    regional_features = []

    for i in range(splits):
        for j in range(splits):

            patch = flows[:, i*h:(i+1)*h, j*w:(j+1)*w, :]

            vx = patch[..., 0]
            vy = patch[..., 1]

            # magnitude
            mag = np.sqrt(vx**2 + vy**2)

            # direction
            angle = np.arctan2(vy, vx)

            regional_features.append({
                "mag": mag.mean(axis=(1,2)),
                "angle": angle.mean(axis=(1,2))
            })

    return regional_features

# =========================
# FEATURE EXTRACTION
# =========================
def compute_features(regional):

    mags = np.array([r["mag"] for r in regional])   # (regions, T)
    angles = np.array([r["angle"] for r in regional])

    eps = 1e-6

    # ---- Global magnitude
    global_mag = np.mean(mags, axis=0)

    # ---- Deviation (motion strength variability)
    deviation = np.std(global_mag) / (np.mean(global_mag) + eps)

    # ---- Spatial variance (heterogeneity)
    spatial_var = np.var(mags, axis=0).mean()
    spatial_var /= (np.mean(global_mag)**2 + eps)

    # ---- Temporal instability
    temporal_instability = np.std(np.diff(global_mag))
    temporal_instability /= (np.mean(global_mag) + eps)

    # =========================
    # NEW FLOW FEATURES
    # =========================

    # ---- Direction consistency (phase coherence)
    direction_consistency = np.abs(np.mean(np.exp(1j * angles)))

    # ---- Divergence (approx)
    divergence = np.mean(np.diff(global_mag))

    # ---- Flow energy
    flow_energy = np.mean(global_mag)

    return deviation, spatial_var, temporal_instability, direction_consistency, divergence, flow_energy

# =========================
# PROCESS PATIENT
# =========================
def process_patient(path):

    data = load_4d(path)
    flows = compute_flow_sequence(data)

    regional = regional_flow_features(flows)

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
            features = process_patient(nii_path)
        except Exception as e:
            print("Skipping:", patient, e)
            continue

        results.append({
            "Patient": patient,
            "Deviation": features[0],
            "SpatialVar": features[1],
            "TemporalInstability": features[2],
            "DirectionConsistency": features[3],
            "Divergence": features[4],
            "FlowEnergy": features[5]
        })

        print("Processed:", patient)

    df = pd.DataFrame(results)

    os.makedirs("outputs_v19", exist_ok=True)
    df.to_csv("outputs_v19/features.csv", index=False)

    print("\nSaved: outputs_v19/features.csv")

    return df

# =========================
# ANALYSIS
# =========================
def analyze(df):

    X = df[[
        "Deviation",
        "SpatialVar",
        "TemporalInstability",
        "DirectionConsistency",
        "Divergence",
        "FlowEnergy"
    ]].values

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
            subset["DirectionConsistency"],
            color=colors[c],
            label=f"Cluster {c}"
        )

    plt.xlabel("Deviation")
    plt.ylabel("DirectionConsistency")
    plt.title("V19 Optical Flow State Space")
    plt.legend()
    plt.savefig("outputs_v19/state_space.png")

    df.to_csv("outputs_v19/results.csv", index=False)

    print("\nSaved: outputs_v19/results.csv")

# =========================
# RUN
# =========================
if __name__ == "__main__":

    df = process_dataset("data")
    analyze(df)
