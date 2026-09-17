import numpy as np


def compute_fvc(x):
    return float((np.max(x) - np.min(x)) / (np.mean(x) + 1e-8))


def compute_attractor_area(traj):

    x = traj[:, 0]
    y = traj[:, 1]

    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return float(area)


def compute_phase_asymmetry(phase):

    phase = np.unwrap(phase)

    mid = len(phase) // 2

    systole = phase[:mid]
    diastole = phase[mid:]

    return float(len(systole) / (len(diastole) + 1e-8))


def compute_phase_velocity_var(phase):

    dtheta = np.diff(phase)
    return float(np.std(dtheta))


def classify_dynamics(fvc, area, asym, vel_var, instability):

    if fvc > 0.5:
        contraction = "STRONG"
    elif fvc > 0.3:
        contraction = "MODERATE"
    else:
        contraction = "WEAK"

    if instability < 0.15:
        stability = "STABLE"
    elif instability < 0.30:
        stability = "MODERATE"
    else:
        stability = "UNSTABLE"

    if asym < 0.8 or asym > 1.2:
        symmetry = "ASYMMETRIC"
    else:
        symmetry = "BALANCED"

    return {
        "Contraction": contraction,
        "Stability": stability,
        "Symmetry": symmetry
    }
