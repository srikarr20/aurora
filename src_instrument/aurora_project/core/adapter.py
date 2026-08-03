import numpy as np
import scipy.ndimage as ndi

class V2Adapter:
    def __init__(self, cx=70, alpha=0.7):
        self.cx = cx
        self.alpha = alpha
        self.prev_mask = None

    def simple_mask(self, frame):
        thresh = np.percentile(frame, self.cx)
        return (frame > thresh).astype(float)

    def stabilize_mask(self, mask):
        if self.prev_mask is None:
            smoothed = mask
        else:
            smoothed = self.alpha * self.prev_mask + (1 - self.alpha) * mask

        smoothed = (smoothed > 0.5).astype(int)

        labeled, num = ndi.label(smoothed)
        if num > 0:
            sizes = ndi.sum(smoothed, labeled, range(1, num+1))
            largest = (labeled == (np.argmax(sizes)+1))
            smoothed = largest.astype(int)

        self.prev_mask = smoothed
        return smoothed

    def to_signal(self, V):
        T = V.shape[3]
        x = []

        for t in range(T):
            frame = V[:,:,:,t]
            mask = self.simple_mask(frame)
            mask = self.stabilize_mask(mask)

            val = np.mean(frame[mask > 0]) if np.sum(mask) > 0 else 0
            x.append(val)

        return np.array(x)
