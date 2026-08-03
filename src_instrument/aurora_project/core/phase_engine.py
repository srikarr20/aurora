import numpy as np
from scipy.signal import hilbert


# -----------------------------
# BUILD TRAJECTORY (FIXED)
# -----------------------------
def build_phase_trajectory(x):

    x = np.array(x)

    analytic = hilbert(x)
    phase = np.angle(analytic)

    # -----------------------------
    # 🔥 AMPLITUDE SCALING (KEY FIX)
    # -----------------------------
    amplitude = (x - np.mean(x)) / (np.std(x) + 1e-8)

    traj = np.column_stack((
        amplitude * np.cos(phase),
        amplitude * np.sin(phase)
    ))

    return traj, phase
