import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("outputs_v17/results.csv")

# =========================
# PREPARE DATA
# =========================
X = df["Deviation"].values.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# CLUSTERING
# =========================
kmeans = KMeans(n_clusters=3, random_state=0)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\n=== CLUSTER CENTERS (scaled) ===")
print(kmeans.cluster_centers_)

print("\n=== CLUSTER COUNTS ===")
print(df["Cluster"].value_counts())

# =========================
# SORTED STRUCTURE (IMPORTANT)
# =========================
df_sorted = df.sort_values("Deviation")

plt.figure()
plt.plot(df_sorted["Deviation"].values)
plt.title("Sorted Deviation (Regime Structure)")
plt.xlabel("Index")
plt.ylabel("Deviation")
plt.savefig("outputs_v17/deviation_sorted.png")

# =========================
# CLUSTER VISUALIZATION
# =========================
plt.figure()

colors = ["green", "orange", "red"]

for c in sorted(df["Cluster"].unique()):
    subset = df[df["Cluster"] == c]
    plt.scatter(
        subset["Deviation"],
        [c]*len(subset),
        color=colors[c],
        label=f"Cluster {c}"
    )

plt.xlabel("Deviation")
plt.ylabel("Cluster")
plt.title("Deviation-based Clusters")
plt.legend()
plt.savefig("outputs_v17/deviation_clusters.png")

# =========================
# SAVE (CRITICAL)
# =========================
os.makedirs("outputs_v17", exist_ok=True)
df.to_csv("outputs_v17/results_clustered.csv", index=False)

print("\nSaved: outputs_v17/results_clustered.csv")
