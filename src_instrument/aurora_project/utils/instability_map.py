import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from ..core.regime_engine import label_to_text


def compute_delta_field(V):
    return np.diff(V, axis=3)


def create_instability_video(V, labels, save_path):

    os.makedirs("results", exist_ok=True)

    dV = compute_delta_field(V)

    T = dV.shape[3]
    z_mid = dV.shape[2] // 2

    fig, ax = plt.subplots()

    ims = []

    for t in range(min(T, len(labels) - 1)):

        frame = dV[:, :, z_mid, t]

        vmax = np.percentile(np.abs(frame), 99) + 1e-6

        im = ax.imshow(
            frame,
            cmap="bwr",
            vmin=-vmax,
            vmax=vmax,
            animated=True
        )

        title = ax.text(
            0.5, 1.02,
            f"{label_to_text(labels[t+1])} (Δt={t})",
            ha="center",
            transform=ax.transAxes
        )

        ims.append([im, title])

    ani = animation.ArtistAnimation(fig, ims, interval=60, blit=True)

    print("🎬 Saving instability video...")

    try:
        ani.save(save_path)
        print(f"📸 Saved → {save_path}")
    except Exception as e:
        print("❌ Failed:", e)

    plt.close()
