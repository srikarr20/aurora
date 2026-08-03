import numpy as np


# -----------------------------
# MOTION-AWARE SIGNAL
# -----------------------------
def extract_intensity_signal(V):

    T = V.shape[3]
    signal = []

    for t in range(1, T):

        prev = V[..., t-1]
        curr = V[..., t]

        # motion field
        diff = np.abs(curr - prev)

        # focus on strongest motion
        thresh = np.percentile(diff, 85)
        mask = diff > thresh

        value = np.sum(diff * mask)
        signal.append(value)

    signal = np.array(signal)

    # scale (preserve variation)
    signal = signal / (np.std(signal) + 1e-8)

    return signal
