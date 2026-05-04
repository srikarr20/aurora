print("🔥 AURORA MVP (NO ML)")

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import nibabel as nib


# =========================
# LOAD MRI
# =========================
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return nib.load(path).get_fdata()


# =========================
# NORMALIZE
# =========================
def normalize(V):
    return np.clip((V - np.mean(V)) / (np.std(V) + 1e-8), -5, 5)


# =========================
# RHO
# =========================
def compute_rho(V, cx=70, ct=2):
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
    return (rho - np.min(rho)) / (np.max(rho) + 1e-8)


# =========================
# TRAJECTORY
# =========================
def compute_trajectory(rho, w=5):
    traj = []
    for i in range(len(rho)-w):
        seg = rho[i:i+w]
        traj.append([
            np.mean(seg),
            np.std(seg)
        ])
    return np.array(traj)


# =========================
# METRICS
# =========================
def extract_metrics(rho, traj):
    return {
        "FVC": float(np.max(rho) - np.min(rho)),
        "instability_index": float(np.mean(np.abs(np.diff(rho)))),
        "coherence": float(np.mean(traj[:,0])) if len(traj)>0 else 0,
        "variability": float(np.mean(traj[:,1])) if len(traj)>0 else 0
    }


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


# =========================
# RISK
# =========================
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
# SAVE SIGNAL
# =========================
def save_signal(rho, out):
    plt.figure()
    plt.plot(rho)
    plt.title("ρ(t)")
    plt.savefig(os.path.join(out, "signal.png"))
    plt.close()


# =========================
# SAVE TRAJECTORY
# =========================
def save_trajectory(traj, out):
    if len(traj) == 0:
        return
    plt.figure()
    plt.plot(traj[:,0], traj[:,1])
    plt.xlabel("Coherence")
    plt.ylabel("Variability")
    plt.title("Trajectory")
    plt.savefig(os.path.join(out, "trajectory.png"))
    plt.close()


# =========================
# VIDEO (MRI)
# =========================
def save_mri_video(V, out, rho):
    path = os.path.join(out, "mri_video.mp4")

    h, w = V.shape[0], V.shape[1]
    z = V.shape[2]//2
    T = V.shape[3]

    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (w,h))
    peak = np.argmax(rho)

    for t in range(T):
        frame = V[:,:,z,t]
        frame = (frame-frame.min())/(frame.max()+1e-8)
        frame = (frame*255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if t < len(rho):
            cv2.putText(frame,f"rho:{rho[t]:.2f}",(10,30),0,0.6,(0,255,0),2)

        if t == peak:
            cv2.putText(frame,"PEAK",(10,60),0,0.7,(0,0,255),2)

        video.write(frame)

    video.release()


# =========================
# VIDEO (OVERLAY)
# =========================
def save_overlay(V, R, out):
    path = os.path.join(out, "risk_overlay.mp4")

    h, w = V.shape[0], V.shape[1]
    z = V.shape[2]//2
    T = V.shape[3]

    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (w,h))

    for t in range(T):
        base = V[:,:,z,t]
        base = (base-base.min())/(base.max()+1e-8)
        base = (base*255).astype(np.uint8)
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        risk = R[:,:,z,t]
        if np.max(risk)>0:
            risk = risk/(np.max(risk)+1e-8)

        heat = cv2.applyColorMap((risk*255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(base,0.7,heat,0.6,0)

        video.write(overlay)

    video.release()


# =========================
# MAIN
# =========================
def run_aurora(input_path, out="aurora_results"):

    print("🚀 AURORA MVP START")

    os.makedirs(out, exist_ok=True)

    V = load_data(input_path)
    V = normalize(V)

    rho = compute_rho(V)
    traj = compute_trajectory(rho)

    metrics = extract_metrics(rho, traj)

    M = compute_motion(V)
    R = fuse_risk(M, rho)

    save_signal(rho, out)
    save_trajectory(traj, out)

    save_mri_video(V, out, rho)
    save_overlay(V, R, out)

    with open(os.path.join(out,"report.json"),"w") as f:
        json.dump(metrics,f,indent=4)

    print("🫀 METRICS:")
    print(json.dumps(metrics,indent=2))

    print("✅ MVP COMPLETE")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="aurora_results")
    args = parser.parse_args()

    run_aurora(args.input, args.output)
