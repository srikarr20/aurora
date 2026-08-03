import numpy as np


# =============================
# HANKEL MATRIX
# =============================
def build_hankel(x, window):
    return np.array([x[i:i+window] for i in range(len(x)-window)])


# =============================
# LOCAL SVD (SLIDING)
# =============================
def compute_local_svd(x, window=20, step=1):

    coherences = []
    spectra = []

    for i in range(0, len(x) - window, step):
        segment = x[i:i+window]

        H = build_hankel(segment, window//2)

        if H.shape[0] < 2:
            continue

        _, S, _ = np.linalg.svd(H, full_matrices=False)

        S_norm = S / (np.sum(S) + 1e-8)

        coherences.append(S_norm[0])
        spectra.append(S_norm[:3])  # top 3 modes

    return np.array(coherences), np.array(spectra)


# =============================
# EFFORT (LOCAL)
# =============================
def compute_effort_local(x, window=20):

    dx = np.gradient(x)
    effort = []

    for i in range(0, len(x)-window):
        effort.append(np.mean(np.abs(dx[i:i+window])))

    return np.array(effort)


# =============================
# TRAJECTORY (TIME-RESOLVED)
# =============================
def build_trajectory(coh, spec, effort):

    L = min(len(coh), len(effort), len(spec))

    traj = []

    for i in range(L):
        mode1 = spec[i,0]
        mode2 = spec[i,1] if spec.shape[1] > 1 else 0

        traj.append([mode1, mode2, effort[i]])

    return np.array(traj)


# =============================
# MAIN FEATURE BUILDER
# =============================
def build_features(x):

    coh, spec = compute_local_svd(x)

    effort = compute_effort_local(x)

    traj = build_trajectory(coh, spec, effort)

    return {
        "coherence_ts": coh,
        "mode_spectrum_ts": spec,
        "effort_ts": effort,
        "trajectory": traj,
        "final_state": traj[-1] if len(traj)>0 else [0,0,0]
    }
