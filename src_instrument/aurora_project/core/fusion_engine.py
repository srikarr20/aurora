import numpy as np
from scipy.ndimage import gaussian_filter


# -----------------------------
# NORMALIZE
# -----------------------------
def normalize(x):
    return x / (np.max(x) + 1e-8)


# -----------------------------
# MOTION FIELD
# -----------------------------
def compute_motion_field(V):
    T = V.shape[3]
    M = np.zeros_like(V)

    for t in range(1, T):
        diff = np.abs(V[..., t] - V[..., t-1])
        diff = normalize(diff)
        M[..., t] = diff

    return M


# -----------------------------
# FUSION ENGINE (V3.4.2 FINAL)
# -----------------------------
def fuse_risk(M, instability, V, phase=None):

    T = M.shape[3]

    # Align instability length
    if len(instability) < T:
        instability = np.pad(instability, (0, T - len(instability)), mode='edge')

    inst_norm = normalize(instability)

    # Motion-based weighting
    motion_energy = np.mean(M, axis=3)
    motion_energy = normalize(motion_energy)

    # Consistency map
    consistency = np.std(M, axis=3)
    consistency = normalize(consistency)

    R = np.zeros_like(M)

    for t in range(1, T):

        # -----------------------------
        # BASE
        # -----------------------------
        weight = inst_norm[t-1]
        Rt = M[..., t] * weight

        # -----------------------------
        # SMOOTH FIRST
        # -----------------------------
        Rt = gaussian_filter(Rt, sigma=1.5)

        # -----------------------------
        # SOFT THRESHOLD
        # -----------------------------
        Rt = Rt * (Rt > 0.1 * np.max(Rt))

        # -----------------------------
        # SMOOTH ANATOMICAL MASK
        # -----------------------------
        p = np.percentile(V[..., t], 70)
        mask = V[..., t] > p
        mask = gaussian_filter(mask.astype(float), sigma=2) > 0.5
        Rt *= mask

        # -----------------------------
        # CONSISTENCY
        # -----------------------------
        Rt *= consistency

        # -----------------------------
        # MOTION WEIGHTING
        # -----------------------------
        Rt *= motion_energy

        # -----------------------------
        # PHASE GATING
        # -----------------------------
        if phase is not None and len(phase) > t:
            phase_weight = np.sin(2 * np.pi * phase[t])**2
            Rt *= phase_weight

        Rt = np.nan_to_num(Rt)

        R[..., t] = Rt

    return R


# -----------------------------
# GLOBAL MAP
# -----------------------------
def compute_global_risk(R):
    R_total = np.mean(R, axis=3)
    return normalize(R_total)
