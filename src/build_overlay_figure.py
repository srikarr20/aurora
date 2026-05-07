import os
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
# DELTA V
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
# BUILD TRAJECTORIES
# ============================================

results = []

for name, path in CASES:

    print("Processing:", name)

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

    results.append({
        "name": name,
        "X": X
    })

# ============================================
# OVERLAY FIGURE
# ============================================

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(
    111,
    projection='3d'
)

# --------------------------------------------
# PLOT
# --------------------------------------------

for r in results:

    X = r["X"]

    ax.plot(
        X[:,0],
        X[:,1],
        X[:,2],
        linewidth=3,
        label=r["name"]
    )

    ax.scatter(
        X[-1,0],
        X[-1,1],
        X[-1,2],
        s=100
    )

# --------------------------------------------
# AXES
# --------------------------------------------

lim = 3

ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)

ax.set_xlabel("C")
ax.set_ylabel("K")
ax.set_zlabel("E")

ax.set_title(
    "Differential Manifold Trajectory Overlay",
    fontsize=16
)

ax.legend()

# ============================================
# SAVE
# ============================================

plt.tight_layout()

out_path = os.path.join(
    OUT_DIR,
    "figure_2_trajectory_overlay.png"
)

plt.savefig(
    out_path,
    dpi=300
)

print("\nSAVED:")
print(out_path)
