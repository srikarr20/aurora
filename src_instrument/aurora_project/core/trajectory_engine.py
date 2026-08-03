import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# COMPUTE TRAJECTORY
# -----------------------------
def compute_trajectory(Xp_sequence):

    Xp_sequence = np.array(Xp_sequence)

    velocities = np.diff(Xp_sequence, axis=0)
    speed = np.linalg.norm(velocities, axis=1)

    drift = float(np.mean(speed))

    return velocities, speed, drift


# -----------------------------
# PROJECT CENTERS INTO PCA SPACE
# -----------------------------
def project_centers_to_pca(centers, pca):

    return pca.transform(centers)


# -----------------------------
# DIRECTION (FIXED)
# -----------------------------
def compute_direction(current_point, centers_pca):

    distances = []

    for c in centers_pca:
        d = np.linalg.norm(current_point - c)
        distances.append(d)

    target_cluster = int(np.argmin(distances))

    return target_cluster


# -----------------------------
# DRIFT RISK (UPDATED SCALE)
# -----------------------------
def compute_drift_risk(drift):

    # your drift is larger → rescale thresholds
    if drift < 0.5:
        return "STABLE"
    elif drift < 1.5:
        return "MODERATE DRIFT"
    else:
        return "HIGH DRIFT"


# -----------------------------
# PLOT TRAJECTORY
# -----------------------------
def plot_trajectory(Xp_sequence, out_path):

    Xp_sequence = np.array(Xp_sequence)

    plt.figure(figsize=(6,6))

    plt.plot(Xp_sequence[:,0], Xp_sequence[:,1], '-o')

    for i, (x, y) in enumerate(Xp_sequence):
        plt.text(x, y, f"t{i}", fontsize=8)

    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("Patient Trajectory in Phenotype Space")
    plt.grid(True)

    plt.savefig(out_path)
    plt.close()
