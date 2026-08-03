import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from ..core.regime_engine import label_to_text


# =========================================
# TEMPORAL RISK (SLIDING WINDOW)
# =========================================
def compute_temporal_risk(V, window=5):

    T = V.shape[3]
    risk_t = []

    for t in range(window, T):

        segment = V[:, :, :, t-window:t]

        var_map = np.var(segment, axis=3)
        dV = np.diff(segment, axis=3)
        grad_map = np.mean(np.abs(dV), axis=3)

        risk = var_map + grad_map

        # normalize per frame
        risk = (risk - risk.min()) / (risk.max() + 1e-8)

        risk_t.append(risk)

    return np.stack(risk_t, axis=3)


# =========================================
# VIDEO
# =========================================
def create_risk_evolution_video(V, labels, save_path):

    os.makedirs("results", exist_ok=True)

    risk_t = compute_temporal_risk(V)

    T = risk_t.shape[3]
    z_mid = risk_t.shape[2] // 2

    fig, ax = plt.subplots()

    ims = []

    for t in range(min(T, len(labels) - 5)):

        frame = risk_t[:, :, z_mid, t]

        im = ax.imshow(
            frame,
            cmap="hot",
            vmin=0,
            vmax=1,
            animated=True
        )

        title = ax.text(
            0.5, 1.02,
            f"{label_to_text(labels[t+5])} (risk buildup)",
            ha="center",
            transform=ax.transAxes
        )

        ims.append([im, title])

    ani = animation.ArtistAnimation(fig, ims, interval=80, blit=True)

    print("🎬 Saving risk evolution video...")

    try:
        ani.save(save_path)
        print(f"📸 Saved → {save_path}")
    except Exception as e:
        print("❌ Failed:", e)

    plt.close()
