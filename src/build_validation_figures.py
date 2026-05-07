import os
import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# ============================================
# CASES
# ============================================

CASES = [
    (
        "STABLE",
        "examples/stable/patient029_4d.nii.gz"
    ),
    (
        "IRREGULAR",
        "examples/irregular/patient094_4d.nii.gz"
    ),
    (
        "LOW CONTRACTION",
        "examples/low_contraction/patient008_4d.nii.gz"
    )
]

# ============================================
# OUTPUT
# ============================================

OUT_DIR = "outputs/figure_suite"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================
# LOAD
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
# LOAD ALL CASES
# ============================================

results = []

for name, path in CASES:

    print("Loading:", name)

    data = load_4d(path)

    deltaV = compute_deltaV(data)

    C, K, E = compute_observables(deltaV)

    Cn = norm(C)
    Kn = norm(K)
    En = norm(E)

    X = np.vstack([
        Cn,
        Kn,
        En
    ]).T

    # accumulation map
    z = data.shape[2] // 2

    accum = np.sum(
        deltaV[:, :, z, :],
        axis=2
    )

    accum = accum / (
        accum.max() + 1e-6
    )

    results.append({
        "name": name,
        "C": Cn,
        "K": Kn,
        "E": En,
        "X": X,
        "accum": accum
    })

# ============================================
# FIGURE 1
# ============================================

fig = plt.figure(
    figsize=(18, 12)
)

# ============================================
# ROW 1 — ACCUMULATION MAPS
# ============================================

for i, r in enumerate(results):

    ax = fig.add_subplot(
        3,
        3,
        i + 1
    )

    ax.imshow(
        r["accum"],
        cmap="hot"
    )

    ax.set_title(
        r["name"],
        fontsize=14
    )

    ax.axis("off")

# ============================================
# ROW 2 — OBSERVABLE EVOLUTION
# ============================================

for i, r in enumerate(results):

    ax = fig.add_subplot(
        3,
        3,
        i + 4
    )

    ax.plot(
        r["C"],
        label="C"
    )

    ax.plot(
        r["K"],
        label="K"
    )

    ax.plot(
        r["E"],
        label="E"
    )

    ax.set_ylim(-3, 3)

    ax.set_title(
        f'{r["name"]} Observables'
    )

    ax.legend()

# ============================================
# ROW 3 — STATE SPACE
# ============================================

lim = 3

for i, r in enumerate(results):

    ax = fig.add_subplot(
        3,
        3,
        i + 7,
        projection='3d'
    )

    X = r["X"]

    ax.plot(
        X[:,0],
        X[:,1],
        X[:,2],
        linewidth=2
    )

    ax.scatter(
        X[-1,0],
        X[-1,1],
        X[-1,2],
        s=80
    )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.set_xlabel("C")
    ax.set_ylabel("K")
    ax.set_zlabel("E")

    ax.set_title(
        f'{r["name"]} State Space'
    )

# ============================================
# SAVE
# ============================================

plt.tight_layout()

out_path = os.path.join(
    OUT_DIR,
    "figure_1_regime_comparison.png"
)

plt.savefig(
    out_path,
    dpi=300
)

print("\nSAVED:")
print(out_path)
