import numpy as np


# ----------------------------------------
# FIT IDEAL CIRCLE (REFERENCE ATTRACTOR)
# ----------------------------------------
def fit_reference_circle(traj):
    """
    Estimate center + radius of attractor
    """
    center = np.mean(traj, axis=0)
    radii = np.linalg.norm(traj - center, axis=1)
    radius = np.mean(radii)

    return center, radius


# ----------------------------------------
# DISTORTION SCORE
# ----------------------------------------
def compute_attractor_distortion(traj, center, radius):
    """
    Measures how much trajectory deviates from ideal circle
    """
    radii = np.linalg.norm(traj - center, axis=1)
    deviation = np.abs(radii - radius)

    score = np.mean(deviation)
    return score, deviation


# ----------------------------------------
# LOCAL INSTABILITY
# ----------------------------------------
def compute_local_instability(traj):
    """
    Measures local velocity / irregularity
    """
    velocity = np.linalg.norm(np.diff(traj, axis=0), axis=1)

    # pad to match length
    velocity = np.concatenate([[velocity[0]], velocity])

    return velocity
