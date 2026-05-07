import os
import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ============================================
# CASES
# ============================================

CASES = [
    (
        "stable",
        "examples/stable/patient029_4d.nii.gz"
    ),
    (
        "irregular",
        "examples/irregular/patient094_4d.nii.gz"
    ),
    (
        "low_contraction",
        "examples/low_contraction/patient008_4d.nii.gz"
    )
]

# ============================================
# OUTPUT
# ============================================

OUT_DIR = "outputs/validation_videos"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_W = 256
TARGET_H = 256

FPS = 5

# ============================================
# LOAD MRI
# ============================================

def load_4d(path):

    data = np.squeeze(
        nib.load(path).get_fdata()
    )

    data = (
        data - data.min()
    ) / (
        data.max() - data.min() + 1e-8
    )

    return data

# ============================================
# DETECTOR
# ============================================

def compute_deltaV(data):

    return np.abs(
        np.diff(data, axis=3)
    )

# ============================================
# OBSERVABLES
# ============================================

def compute_observables(deltaV):

    T = deltaV.shape[3]

    C = []
    K = []
    E = []

    for t in range(T):

        frame = deltaV[:, :, :, t]

        c = np.std(frame) / (
            np.mean(frame) + 1e-6
        )

        if t > 0:
            k = np.std(
                frame - deltaV[:, :, :, t-1]
            )
        else:
            k = 0

        e = np.sum(frame)

        C.append(c)
        K.append(k)
        E.append(e)

    return (
        np.array(C),
        np.array(K),
        np.array(E)
    )

# ============================================
# NORMALIZE
# ============================================

def norm(x):

    return (
        x - np.mean(x)
    ) / (
        np.std(x) + 1e-6
    )

# ============================================
# RENDER CASE
# ============================================

def render_case(case_name, input_path):

    print("\n===================================")
    print("CASE:", case_name)
    print("===================================")

    # ----------------------------------------
    # LOAD
    # ----------------------------------------

    data = load_4d(input_path)

    H, W, Z, T_full = data.shape

    z = Z // 2

    deltaV = compute_deltaV(data)

    T = deltaV.shape[3]

    C, K, E = compute_observables(deltaV)

    Cn = norm(C)
    Kn = norm(K)
    En = norm(E)

    X = np.vstack([
        Cn,
        Kn,
        En
    ]).T

    # ----------------------------------------
    # FIXED AXIS LIMITS
    # ----------------------------------------

    lim = 3

    # ----------------------------------------
    # OUTPUT
    # ----------------------------------------

    output_video = os.path.join(
        OUT_DIR,
        f"{case_name}_validation.mp4"
    )

    canvas_w = TARGET_W * 3
    canvas_h = TARGET_H * 2

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        FPS,
        (canvas_w, canvas_h)
    )

    print("Rendering:", output_video)

    # ----------------------------------------
    # LOOP
    # ----------------------------------------

    accum = np.zeros((H, W))

    for t in range(T):

        # ====================================
        # RAW MRI
        # ====================================

        raw = data[:, :, z, t]

        raw_img = cv2.resize(
            raw,
            (TARGET_W, TARGET_H)
        )

        raw_img = (
            raw_img * 255
        ).astype(np.uint8)

        raw_img = cv2.cvtColor(
            raw_img,
            cv2.COLOR_GRAY2BGR
        )

        cv2.putText(
            raw_img,
            "RAW MRI",
            (10,25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        # ====================================
        # DELTA V
        # ====================================

        dv = deltaV[:, :, z, t]

        if dv.max() > 0:
            dv_norm = dv / dv.max()
        else:
            dv_norm = dv

        dv_img = cv2.resize(
            dv_norm,
            (TARGET_W, TARGET_H)
        )

        dv_img = cv2.applyColorMap(
            (dv_img * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        cv2.putText(
            dv_img,
            "DELTA V",
            (10,25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        # ====================================
        # ACCUMULATION
        # ====================================

        accum += dv

        acc = accum / (
            accum.max() + 1e-6
        )

        acc_img = cv2.resize(
            acc,
            (TARGET_W, TARGET_H)
        )

        acc_img = cv2.applyColorMap(
            (acc_img * 255).astype(np.uint8),
            cv2.COLORMAP_HOT
        )

        cv2.putText(
            acc_img,
            "ACCUMULATION",
            (10,25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        # ====================================
        # OBSERVABLE EVOLUTION
        # ====================================

        fig, ax = plt.subplots(
            figsize=(4,4)
        )

        ax.plot(
            Cn[:t+1],
            label="C"
        )

        ax.plot(
            Kn[:t+1],
            label="K"
        )

        ax.plot(
            En[:t+1],
            label="E"
        )

        ax.set_ylim(-3, 3)

        ax.set_title(
            "Observable Evolution"
        )

        ax.legend()

        fig.canvas.draw()

        buf = np.asarray(
            fig.canvas.buffer_rgba()
        )

        metrics_img = cv2.cvtColor(
            buf,
            cv2.COLOR_RGBA2RGB
        )

        plt.close(fig)

        metrics_img = cv2.resize(
            metrics_img,
            (TARGET_W, TARGET_H)
        )

        # ====================================
        # STATE SPACE
        # ====================================

        fig = plt.figure(
            figsize=(4,4)
        )

        ax = fig.add_subplot(
            111,
            projection='3d'
        )

        ax.plot(
            X[:t+1,0],
            X[:t+1,1],
            X[:t+1,2],
            linewidth=2
        )

        ax.scatter(
            X[t,0],
            X[t,1],
            X[t,2],
            s=80
        )

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

        ax.set_xlabel("C")
        ax.set_ylabel("K")
        ax.set_zlabel("E")

        ax.set_title(
            "State Space"
        )

        fig.canvas.draw()

        buf = np.asarray(
            fig.canvas.buffer_rgba()
        )

        state_img = cv2.cvtColor(
            buf,
            cv2.COLOR_RGBA2RGB
        )

        plt.close(fig)

        state_img = cv2.resize(
            state_img,
            (TARGET_W, TARGET_H)
        )

        # ====================================
        # INFO PANEL
        # ====================================

        info = np.zeros(
            (TARGET_H, TARGET_W, 3),
            dtype=np.uint8
        )

        cv2.putText(
            info,
            f"CASE: {case_name}",
            (20,35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.putText(
            info,
            f"Frame: {t}",
            (20,75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

        cv2.putText(
            info,
            f"C: {Cn[t]:.3f}",
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )

        cv2.putText(
            info,
            f"K: {Kn[t]:.3f}",
            (20,165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.putText(
            info,
            f"E: {En[t]:.3f}",
            (20,210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,0,255),
            2
        )

        # ====================================
        # COMBINE
        # ====================================

        top = np.hstack([
            raw_img,
            dv_img,
            acc_img
        ])

        bottom = np.hstack([
            metrics_img,
            state_img,
            info
        ])

        canvas = np.vstack([
            top,
            bottom
        ])

        writer.write(canvas)

    writer.release()

    print("DONE:", output_video)

# ============================================
# MAIN
# ============================================

for case_name, input_path in CASES:

    render_case(
        case_name,
        input_path
    )

print("\nALL VIDEOS GENERATED")
