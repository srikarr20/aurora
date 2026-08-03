import nibabel as nib
import numpy as np
import os
import re


class ACDCLVAdapter:
    """
    ACDC LV Adapter (FINAL)

    - Loads 4D MRI
    - Detects frame-wise GT files
    - Builds full LV volume signal
    """

    def __init__(self, lv_label=3):
        self.lv_label = lv_label

    # ----------------------------------------
    # LOAD MRI + GT
    # ----------------------------------------
    def load(self, data_file):

        nii = nib.load(data_file)
        V = nii.get_fdata()   # (x, y, z, t)

        base_dir = os.path.dirname(data_file)

        gt_dict = {}

        for f in os.listdir(base_dir):

            if f.endswith("_gt.nii.gz") or f.endswith("_gt.nii"):

                match = re.search(r'frame(\d+)', f)

                if match:
                    frame_idx = int(match.group(1)) - 1  # zero-based

                    gt_path = os.path.join(base_dir, f)
                    gt = nib.load(gt_path).get_fdata()

                    gt_dict[frame_idx] = gt

        if len(gt_dict) == 0:
            raise ValueError("No GT files found in folder")

        return V, gt_dict

    # ----------------------------------------
    # BUILD FULL LV SIGNAL
    # ----------------------------------------
    def to_signal(self, V, gt_dict):

        T = V.shape[3]
        lv_volume = np.zeros(T)

        # fill known GT frames
        for t, seg in gt_dict.items():
            lv_mask = (seg == self.lv_label)
            lv_volume[t] = np.sum(lv_mask)

        # sort known frames
        known_idx = sorted(gt_dict.keys())

        # interpolate between known frames
        for i in range(len(known_idx) - 1):

            t1 = known_idx[i]
            t2 = known_idx[i + 1]

            v1 = lv_volume[t1]
            v2 = lv_volume[t2]

            for t in range(t1 + 1, t2):
                alpha = (t - t1) / (t2 - t1)
                lv_volume[t] = (1 - alpha) * v1 + alpha * v2

        # circular interpolation (end → start)
        t1 = known_idx[-1]
        t2 = known_idx[0] + T

        v1 = lv_volume[t1]
        v2 = lv_volume[known_idx[0]]

        for t in range(t1 + 1, T):
            alpha = (t - t1) / (t2 - t1)
            lv_volume[t] = (1 - alpha) * v1 + alpha * v2

        # normalize
        lv_volume = (lv_volume - lv_volume.min()) / (lv_volume.max() + 1e-8)

        return lv_volume
