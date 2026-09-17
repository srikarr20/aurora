import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from ..core.regime_engine import label_to_text


def compute_edges(frame):
    gx = np.gradient(frame, axis=0)
    gy = np.gradient(frame, axis=1)
    return np.sqrt(gx**2 + gy**2)


def create_overlay_video(V, labels, save_path):

    os.makedirs("results", exist_ok=True)

    T = V.shape[3]
    z_mid = V.shape[2] // 2

    fig, ax = plt.subplots()

    ims = []

    for t in range(min(T, len(labels))):

        frame = V[:, :, z_mid, t]

        # ✅ ORIGINAL FIELD (keep strong contrast)
        base = ax.imshow(frame, animated=True, cmap="viridis")

        # 🔥 EDGE MAP (this is the real signal)
        edges = compute_edges(frame)

        if labels[t] == 0:
            cmap = "Greens"
            alpha = 0.2
        elif labels[t] == 1:
            cmap = "plasma"
            alpha = 0.35
        else:
            cmap = "inferno"
            alpha = 0.5

        edge_overlay = ax.imshow(
            edges,
            cmap=cmap,
            alpha=alpha,
            animated=True
        )

        title = ax.text(
            0.5, 1.02,
            f"{label_to_text(labels[t])} (t={t})",
            ha="center",
            transform=ax.transAxes,
            fontsize=12
        )

        ims.append([base, edge_overlay, title])

    ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True)

    print("🎬 Saving improved overlay video...")

    try:
        ani.save(save_path)
        print(f"📸 Saved overlay → {save_path}")
    except Exception as e:
        print("❌ Failed:", e)

    plt.close()
