import numpy as np


# -----------------------------
# BUILD BASELINE
# -----------------------------
def build_baseline(metrics_list):

    keys = metrics_list[0].keys()

    baseline = {}

    for k in keys:
        values = np.array([m[k] for m in metrics_list])
        baseline[k] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values) + 1e-8)
        }

    return baseline


# -----------------------------
# COMPUTE Z-SCORES
# -----------------------------
def compute_deviation(metrics, baseline):

    zscores = {}

    for k in metrics:
        mu = baseline[k]["mean"]
        sigma = baseline[k]["std"]

        z = (metrics[k] - mu) / sigma
        zscores[k] = float(z)

    return zscores


# -----------------------------
# CLASSIFY ANOMALY
# -----------------------------
def classify_anomaly(zscores):

    score = 0

    for k, z in zscores.items():
        score += abs(z)

    score /= len(zscores)

    if score < 1:
        return "NORMAL"
    elif score < 2:
        return "MILD DEVIATION"
    else:
        return "ABNORMAL"
