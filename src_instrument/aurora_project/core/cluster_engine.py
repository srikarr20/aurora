import numpy as np
from sklearn.cluster import KMeans


# -----------------------------
# BUILD FEATURE MATRIX
# -----------------------------
def build_feature_matrix(metrics_list, keys=None):

    if keys is None:
        keys = ["FVC", "Area", "PhaseVar", "Instability"]

    X = np.array([[m[k] for k in keys] for m in metrics_list])

    return X, keys


# -----------------------------
# NORMALIZE FEATURES
# -----------------------------
def normalize_features(X):

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0) + 1e-8

    Xn = (X - mu) / sigma

    return Xn, mu, sigma


# -----------------------------
# CLUSTER
# -----------------------------
def cluster_patients(Xn, k=3, random_state=42):

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(Xn)

    centers = model.cluster_centers_

    return labels, centers, model


# -----------------------------
# PHENOTYPE NAMING (FIXED)
# -----------------------------
def name_clusters(centers, keys):

    names = []

    for c in centers:

        fvc = c[keys.index("FVC")]
        inst = c[keys.index("Instability")]
        area = c[keys.index("Area")]
        phase = c[keys.index("PhaseVar")]

        # -----------------------------
        # Contraction
        # -----------------------------
        if fvc > 0:
            contraction = "Strong"
        else:
            contraction = "Weak"

        # -----------------------------
        # Stability
        # -----------------------------
        if inst < 0:
            stability = "Stable"
        else:
            stability = "Unstable"

        # -----------------------------
        # Geometry
        # -----------------------------
        if area > 0:
            geometry = "Expansive"
        else:
            geometry = "Compact"

        # -----------------------------
        # Dynamics
        # -----------------------------
        if phase > 0:
            dynamics = "Dynamic"
        else:
            dynamics = "Calm"

        name = f"{contraction} | {stability} | {geometry} | {dynamics}"
        names.append(name)

    return names


# -----------------------------
# ASSIGN PHENOTYPES
# -----------------------------
def assign_phenotypes(labels, cluster_names):

    return [cluster_names[l] for l in labels]
