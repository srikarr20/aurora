import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import nibabel as nib
from scipy.signal import butter, filtfilt


# =========================
# LOAD + NORMALIZE
# =========================
def load_data(path):
    return nib.load(path).get_fdata()

def normalize(V):
    return np.clip((V - np.mean(V)) / (np.std(V) + 1e-8), -5, 5)


# =========================
# SAFE FILTER (ONLY FOR FVC)
# =========================
def bandpass_filter(signal):
    if len(signal) < 10:
        return signal

    try:
        b, a = butter(2, [0.05, 0.4], btype='band')
        return filtfilt(b, a, signal)
    except:
        return signal


# =========================
# RAW RHO (NO FILTER)
# =========================
def compute_rho_raw(V, cx=70, ct=2):
    rho = []

    for t in range(ct, V.shape[-1]):
        prev = V[..., t-ct]
        curr = V[..., t]

        mask = curr > np.percentile(curr, cx)

        if np.sum(mask) < 500:
            rho.append(0)
            continue

        diff = np.abs(curr - prev)
        val = np.mean(diff[mask]) / (np.mean(np.abs(curr[mask])) + 1e-8)
        rho.append(val)

    rho = np.array(rho)

    if len(rho) > 2:
        rho = rho[1:-1]

    # normalize
    rho = (rho - np.min(rho)) / (np.max(rho) + 1e-8)

    return rho


# =========================
# FILTERED RHO (FOR FVC)
# =========================
def compute_rho_filtered(rho_raw):
    rho = bandpass_filter(rho_raw)

    # smooth
    rho = np.convolve(rho, np.ones(3)/3, mode='same')

    # clamp
    if len(rho) > 5:
        rho = np.clip(rho,
                      np.percentile(rho, 5),
                      np.percentile(rho, 95))

    # normalize again
    rho = (rho - np.min(rho)) / (np.max(rho) + 1e-8)

    return rho


# =========================
# TRAJECTORY
# =========================
def compute_trajectory(rho, w=5):
    traj = []
    for i in range(len(rho)-w):
        seg = rho[i:i+w]
        traj.append([np.mean(seg), np.std(seg)])
    return np.array(traj)


# =========================
# METRICS (FIXED)
# =========================
def extract_metrics(rho_raw, rho_filtered, traj):

    instability = np.mean(np.abs(np.diff(rho_raw)))  # IMPORTANT

    return {
        "FVC": float(np.max(rho_filtered) - np.min(rho_filtered)),
        "instability_index": float(instability),
        "coherence": float(np.mean(traj[:,0])) if len(traj)>0 else 0,
        "variability": float(np.mean(traj[:,1])) if len(traj)>0 else 0,
    }


# =========================
# CLASSIFIER
# =========================
def classify_behavior(metrics):
    if metrics["instability_index"] > 0.07:
        return "IRREGULAR"
    elif metrics["FVC"] < 0.47:
        return "LOW_CONTRACTION"
    else:
        return "STABLE"


# =========================
# MOTION
# =========================
def compute_motion(V):
    M = np.zeros_like(V)
    for t in range(1, V.shape[3]):
        diff = np.abs(V[...,t] - V[...,t-1])
        diff /= (np.max(diff)+1e-8)
        M[...,t] = diff
    return M


def fuse_risk(M, rho):
    R = np.zeros_like(M)
    rho_n = rho / (np.max(rho)+1e-8)

    valid_T = min(M.shape[3], len(rho_n)+1)

    for t in range(1, valid_T):
        Rt = M[...,t] * rho_n[t-1]
        if np.max(Rt) > 0:
            Rt[Rt < 0.2*np.max(Rt)] = 0
        R[...,t] = Rt

    return R


# =========================
# SAVE
# =========================
def save_signal(rho, out):
    plt.figure()
    plt.plot(rho)
    plt.savefig(os.path.join(out, "signal.png"))
    plt.close()


# =========================
# MAIN
# =========================
def run_aurora(input_path, out):

    os.makedirs(out, exist_ok=True)

    V = normalize(load_data(input_path))

    rho_raw = compute_rho_raw(V)
    rho_filtered = compute_rho_filtered(rho_raw)

    traj = compute_trajectory(rho_filtered)

    metrics = extract_metrics(rho_raw, rho_filtered, traj)
    label = classify_behavior(metrics)
    metrics["behavior_label"] = label

    M = compute_motion(V)
    R = fuse_risk(M, rho_filtered)

    save_signal(rho_filtered, out)

    with open(os.path.join(out,"report.json"),"w") as f:
        json.dump(metrics,f,indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_aurora(args.input, args.output)
