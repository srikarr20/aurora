import numpy as np
import matplotlib.pyplot as plt
import os


# =============================
# COMPUTE RISK MAP
# =============================
def compute_risk_map(V):

    # temporal variance (how much a voxel fluctuates)
    var_map = np.var(V, axis=3)

    # temporal gradient magnitude (how fast it changes)
    dV = np.diff(V, axis=3)
    grad_map = np.mean(np.abs(dV), axis=3)

    # combine both
    risk = var_map + grad_map

    # normalize
    risk = (risk - risk.min()) / (risk.max() + 1e-8)

    return risk


# =============================
# SAVE CENTRAL SLICE HEATMAP
# =============================
def save_risk_heatmap(risk, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    z_mid = risk.shape[2] // 2
    slice_2d = risk[:, :, z_mid]

    plt.figure(figsize=(6,6))
    plt.imshow(slice_2d, cmap="hot")
    plt.colorbar(label="Risk")
    plt.title("Spatial Risk Map (Central Slice)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"🔥 Saved risk heatmap → {save_path}")
