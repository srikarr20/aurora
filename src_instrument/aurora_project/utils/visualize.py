import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_aurora(x, features, save=True):

    S = np.array(features["stability_ts"])
    E = np.array(features["effort_ts"])

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(12,8))

    # Signal
    plt.subplot(4,1,1)
    plt.plot(x)
    plt.title("x(t) — Extracted Signal")

    # Features
    plt.subplot(4,1,2)
    plt.plot(S, label="Stability")
    plt.plot(E, label="Effort")
    plt.legend()
    plt.title("Feature Streams")

    # Dynamics
    plt.subplot(4,1,3)
    if len(S) > 1:
        plt.plot(np.gradient(S), label="dS")
    if len(E) > 1:
        plt.plot(np.gradient(E), label="dE")
    plt.legend()
    plt.title("Dynamics")

    # Trajectory
    plt.subplot(4,1,4)
    if len(S) > 0 and len(E) > 0:
        plt.plot(S, E)
    plt.xlabel("Stability")
    plt.ylabel("Effort")
    plt.title("Trajectory")

    plt.tight_layout()

    if save:
        img_path = f"results/aurora_plot_{timestamp}.png"
        plt.savefig(img_path, dpi=200)
        print(f"📸 Saved plot → {img_path}")

        np.save(f"results/signal_{timestamp}.npy", x)
        np.save(f"results/stability_{timestamp}.npy", S)
        np.save(f"results/effort_{timestamp}.npy", E)

    plt.show()
