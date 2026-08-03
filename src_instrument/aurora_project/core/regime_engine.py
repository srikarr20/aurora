import numpy as np


# =============================
# SIMPLE K-MEANS (NO SKLEARN)
# =============================
def kmeans(X, k=3, max_iters=50):

    # initialize random centroids
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx]

    for _ in range(max_iters):

        # assign clusters
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # update centroids
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


# =============================
# REGIME DISCOVERY
# =============================
def discover_regimes(traj, k=3):

    if len(traj) < k:
        return np.zeros(len(traj), dtype=int)

    labels, centroids = kmeans(traj, k)

    # sort clusters by coherence (mode1)
    order = np.argsort(centroids[:, 0])[::-1]

    remap = {}
    for new_label, old_label in enumerate(order):
        remap[old_label] = new_label

    mapped = np.array([remap[l] for l in labels])

    return mapped, centroids


# =============================
# LABEL HELPERS
# =============================
def label_to_text(label):
    return {
        0: "STABLE",
        1: "TRANSITION",
        2: "UNSTABLE"
    }.get(label, "UNKNOWN")


def label_to_color(label):
    return {
        0: "green",
        1: "yellow",
        2: "red"
    }.get(label, "white")
