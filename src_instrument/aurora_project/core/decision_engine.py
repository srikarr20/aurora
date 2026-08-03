import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# -----------------------------
# TRAIN CLASSIFIER
# -----------------------------
def train_classifier(Xp, labels):

    clf = LogisticRegression(multi_class="multinomial", max_iter=1000)
    clf.fit(Xp, labels)

    return clf


# -----------------------------
# CREATE GRID
# -----------------------------
def create_grid(Xp, step=0.05):

    x_min, x_max = Xp[:, 0].min() - 1, Xp[:, 0].max() + 1
    y_min, y_max = Xp[:, 1].min() - 1, Xp[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    return xx, yy, grid


# -----------------------------
# PLOT DECISION + RISK
# -----------------------------
def plot_decision_zones(Xp, labels, patient_names, clf, cluster_names, out_path):

    xx, yy, grid = create_grid(Xp)

    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    # probability for "risk" (distance from confident regions)
    probs = clf.predict_proba(grid)
    uncertainty = 1 - np.max(probs, axis=1)
    uncertainty = uncertainty.reshape(xx.shape)

    plt.figure(figsize=(9, 7))

    # -----------------------------
    # Background zones (clusters)
    # -----------------------------
    plt.contourf(xx, yy, Z, alpha=0.2)

    # -----------------------------
    # Risk heatmap (uncertainty)
    # -----------------------------
    plt.contourf(xx, yy, uncertainty, levels=20, cmap="Reds", alpha=0.3)

    # -----------------------------
    # Scatter points
    # -----------------------------
    for i in np.unique(labels):

        idx = labels == i

        plt.scatter(
            Xp[idx, 0],
            Xp[idx, 1],
            label=f"Cluster {i}: {cluster_names[i]}",
            s=80
        )

    # -----------------------------
    # Labels
    # -----------------------------
    for i, name in enumerate(patient_names):
        plt.text(Xp[i, 0], Xp[i, 1], name, fontsize=8)

    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("AURORA Decision Space + Risk Zones")
    plt.legend()
    plt.grid(True)

    plt.savefig(out_path)
    plt.close()
