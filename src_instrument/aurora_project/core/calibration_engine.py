import numpy as np


# -----------------------------
# COMPUTE GT VOLUME
# -----------------------------
def compute_gt_volume(mask, voxel_volume=1.0):
    return float(np.sum(mask) * voxel_volume)


# -----------------------------
# CALIBRATION (GT-ALIGNED)
# -----------------------------
def calibrate_volume_signal(x, gt_dict, voxel_volume=1.0):

    x = np.array(x)

    # -----------------------------
    # Extract GT frames
    # -----------------------------
    gt_frames = sorted(gt_dict.keys())

    if len(gt_frames) < 2:
        raise ValueError("Need at least ED and ES GT frames")

    volumes = {}
    for f in gt_frames:
        volumes[f] = compute_gt_volume(gt_dict[f], voxel_volume)

    # -----------------------------
    # Identify ED / ES from GT
    # -----------------------------
    ED_frame = max(volumes, key=volumes.get)
    ES_frame = min(volumes, key=volumes.get)

    EDV = volumes[ED_frame]
    ESV = volumes[ES_frame]

    # -----------------------------
    # Normalize signal FIRST
    # -----------------------------
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

    # -----------------------------
    # Scale to physical volume
    # -----------------------------
    volume_t = x_norm * (EDV - ESV) + ESV

    return volume_t, EDV, ESV


# -----------------------------
# TRUE EF
# -----------------------------
def compute_ef_from_volume(volume_t):

    volume_t = np.array(volume_t)

    EDV = np.max(volume_t)
    ESV = np.min(volume_t)

    if EDV < 1e-6:
        return 0.0

    EF = (EDV - ESV) / EDV

    return float(EF)
