import nibabel as nib
import numpy as np


class ACDCAdapter:

    def load(self, file_path):

        nii = nib.load(file_path)
        data = nii.get_fdata()

        return data

    def to_signal(self, V):

        x = V.mean(axis=(0, 1, 2))

        x = (x - x.min()) / (x.max() - x.min() + 1e-8)

        return x
