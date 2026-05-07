import numpy as np
import pandas as pd
import nibabel as nib
import os
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =========================
# LOAD
# =========================
def load_4d(path):
    return np.squeeze(nib.load(path).get_fdata())

# =========================
# OPTICAL FLOW
# =========================
def compute_flow(data):

    H, W, Z, T = data.shape
    flows = []

    for t in range(T - 1):

        f1 = data[:, :, Z//2, t].astype(np.float32)
        f2 = data[:, :, Z//2, t+1].astype(np.float32)

        flow = cv2.calcOpticalFlowFarneback(
            f1, f2, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        flows.append(flow)

    return np.array(flows)  # (T, H, W, 2)

# =========================
# REGIONAL FLOW
# =========================
def regional_flow(flows, splits=4):

    T, H, W, _ = flows.shape
    h = H // splits
    w = W // splits

    regions = []

    for i in range(splits):
        for j in range(splits):

            patch = flows[:, i*h:(i+1)*h, j*w:(j+1)*w, :]

            vx = patch[..., 0]
            vy = patch[..., 1]

            mag = np.sqrt(vx**2 + vy**2)

            regions.append({
                "vx": vx.mean(axis=(1,2)),
                "vy": vy.mean(axis=(1,2)),
                "mag": mag.mean(axis=(1,2))
            })

    return regions

# =========================
# COUPLING (NEW)
# =========================
def compute_coupling(regions):

    signals = np.array([r["mag"] for r in regions])  # (regions, T)

    corr_matrix = np.corrcoef(signals)

    # remove diagonal
    n = corr_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)

    return np.mean(corr_matrix[mask])

# =========================
# DIVERGENCE (NEW)
# =========================
def compute_divergence(flows):

    vx = flows[..., 0]
    vy = flows[..., 1]

    # finite differences
    dvx_dx = np.gradient(vx, axis=2)
    dvy_dy = np.gradient(vy, axis=1)

    div = dvx_dx + dvy_dy

    return div

# =========================
# FEATURES
# =========================
def compute_features(regions, flows):

    mags = np.array([r["mag"] for r in regions])
    vx = np.array([r["vx"] for r in regions])
    vy = np.array([r["vy"] for r in regions])

    eps = 1e-6

    global_mag = np.mean(mags, axis=0)

    # ---- existing
    deviation = np.std(global_mag) / (np.mean(global_mag) + eps)

    spatial_var = np.var(mags, axis=0).mean()
    spatial_var /= (np.mean(global_mag)**2 + eps)

    temporal_instability = np.std(np.diff(global_mag))
    temporal_instability /= (np.mean(global_mag) + eps)

    # ---- direction consistency
    angles = np.arctan2(vy, vx)
    direction_consistency = np.abs(np.mean(np.exp(1j * angles)))

    flow_energy = np.mean(global_mag)

    # =========================
    # NEW (V20)
    # =========================

    coupling = compute_coupling(regions)

    div = compute_divergence(flows)
    divergence_variance = np.var(div)

    return (
        deviation,
        spatial_var,
        temporal_instability,
        direction_consistency,
        flow_energy,
        coupling,
        divergence_variance
    )

# =========================
# PROCESS PATIENT
# =========================
def process_patient(path):

    data = load_4d(path)
    flows = compute_flow(data)
    regions = regional_flow(flows)

    return compute_features(regions, flows)

# =========================
# DATASET
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
            f = process_patient(nii_path)
        except Exception as e:
            print("Skipping:", patient, e)
            continue

        results.append({
            "Patient": patient,
            "Deviation": f[0],
            "SpatialVar": f[1],
            "TemporalInstability": f[2],
            "DirectionConsistency": f[3],
            "FlowEnergy": f[4],
            "Coupling": f[5],
            "DivergenceVar": f[6]
        })

        print("Processed:", patient)

    df = pd.DataFrame(results)

    os.makedirs("outputs_v20", exist_ok=True)
    df.to_csv("outputs_v20/features.csv", index=False)

    print("\nSaved: outputs_v20/features.csv")

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
        "FlowEnergy",
        "Coupling",
        "DivergenceVar"
    ]].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=0)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    print("\n=== CLUSTER COUNTS ===")
    print(df["Cluster"].value_counts())

    # visualization
    plt.figure()

    colors = ["green", "orange", "red"]

    for c in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == c]
        plt.scatter(
            subset["Deviation"],
            subset["Coupling"],
            color=colors[c],
            label=f"Cluster {c}"
        )

    plt.xlabel("Deviation")
    plt.ylabel("Coupling")
    plt.title("V20 Topology-Aware State Space")
    plt.legend()
    plt.savefig("outputs_v20/state_space.png")

    df.to_csv("outputs_v20/results.csv", index=False)

    print("\nSaved: outputs_v20/results.csv")

# =========================
# RUN
# =========================
if __name__ == "__main__":

    df = process_dataset("data")
    analyze(df)
