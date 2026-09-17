import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def create_field_video(V, save_path="results/field_video.mp4"):

    os.makedirs("results", exist_ok=True)

    T = V.shape[3]
    z_mid = V.shape[2] // 2  # middle slice

    fig, ax = plt.subplots()

    frames = []

    for t in range(T):
        frame = V[:, :, z_mid, t]

        im = ax.imshow(frame, animated=True, cmap="viridis")
        ax.set_title(f"t = {t}")

        frames.append([im])

    ani = animation.ArtistAnimation(
        fig,
        frames,
        interval=50,
        blit=True
    )

    print("🎬 Saving video...")

    ani.save(save_path)

    print(f"📸 Saved video → {save_path}")

    plt.close()
