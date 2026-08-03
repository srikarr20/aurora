import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# -----------------------------
# PCA PROJECTION
# -----------------------------
def compute_pca(Xn, n_components=2):

    pca = PCA(n_components=n_components)
    Xp = pca.fit_transform(Xn)

    return Xp, pca


# -----------------------------
# PLOT CLUSTERS
# -----------------------------
def plot_clusters(Xp, labels, patient_names, cluster_names, out_path):

    plt.figure(figsize=(8, 6))

    unique_labels = np.unique(labels)

    for i in unique_labels:

        idx = labels == i

        plt.scatter(
            Xp[idx, 0],
            Xp[idx, 1],
            label=f"Cluster {i}: {cluster_names[i]}",
            s=80
        )

    # annotate points
    for i, name in enumerate(patient_names):
        plt.text(Xp[i, 0], Xp[i, 1], name, fontsize=8)

    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("AURORA Phenotype Space")
    plt.legend()
    plt.grid(True)

    plt.savefig(out_path)
    plt.close()
