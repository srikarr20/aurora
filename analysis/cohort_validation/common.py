from pathlib import Path
import math
import re

import nibabel as nib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


GEOMETRY_FEATURES = [
    "compactness",
    "mean_turn_angle_rad",
    "sharp_turn_fraction",
    "radius_cv",
    "mean_abs_observable_sync",
]

ACQUISITION_FEATURES = [
    "native_phases",
    "Z",
    "log_spatial_voxels",
]


def load_volume(path):
    data = np.squeeze(nib.load(str(path)).get_fdata())

    if data.ndim != 4:
        raise ValueError(f"Expected 4D, got {data.shape}")

    if not np.isfinite(data).all():
        raise ValueError("Nonfinite image values")

    lo = float(data.min())
    hi = float(data.max())

    return (data - lo) / (hi - lo + 1e-8)


def compute_cke(data):
    delta = np.abs(np.diff(data, axis=3))

    C, K, E = [], [], []

    for t in range(delta.shape[3]):
        frame = delta[..., t]

        C.append(
            np.std(frame) /
            (np.mean(frame) + 1e-6)
        )

        if t == 0:
            K.append(0.0)
        else:
            K.append(
                np.std(
                    frame - delta[..., t - 1]
                )
            )

        E.append(np.sum(frame))

    return (
        np.asarray(C, dtype=float),
        np.asarray(K, dtype=float),
        np.asarray(E, dtype=float),
    )


def zscore(x):
    return (
        x - np.mean(x)
    ) / (
        np.std(x) + 1e-6
    )


def safe_corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def trajectory_geometry(C, K, E):
    Cn = zscore(C)
    Kn = zscore(K)
    En = zscore(E)

    X = np.column_stack([Cn, Kn, En])

    centroid = X.mean(axis=0)
    radii = np.linalg.norm(X - centroid, axis=1)

    velocity = np.diff(X, axis=0)

    angles = []

    for a, b in zip(velocity[:-1], velocity[1:]):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if na < 1e-12 or nb < 1e-12:
            continue

        cosine = np.clip(
            np.dot(a, b) / (na * nb),
            -1.0,
            1.0,
        )

        angles.append(math.acos(cosine))

    angles = np.asarray(angles)

    corr_values = np.asarray([
        safe_corr(Cn, Kn),
        safe_corr(Cn, En),
        safe_corr(Kn, En),
    ])

    corr_values = corr_values[
        np.isfinite(corr_values)
    ]

    return {
        "compactness": float(np.mean(radii)),
        "mean_turn_angle_rad": (
            float(np.mean(angles))
            if len(angles)
            else np.nan
        ),
        "sharp_turn_fraction": (
            float(np.mean(angles > np.pi / 2))
            if len(angles)
            else np.nan
        ),
        "radius_cv": float(
            np.std(radii) /
            (np.mean(radii) + 1e-12)
        ),
        "mean_abs_observable_sync": (
            float(np.mean(np.abs(corr_values)))
            if len(corr_values)
            else np.nan
        ),
    }


def residualize_train_test(
    train,
    test,
    feature_names=GEOMETRY_FEATURES,
    acquisition_names=ACQUISITION_FEATURES,
):
    acq_scaler = StandardScaler()

    A_train = acq_scaler.fit_transform(
        train[acquisition_names].to_numpy(float)
    )

    A_test = acq_scaler.transform(
        test[acquisition_names].to_numpy(float)
    )

    train_resid = np.zeros(
        (len(train), len(feature_names))
    )

    test_resid = np.zeros(
        (len(test), len(feature_names))
    )

    for j, feature in enumerate(feature_names):
        reg = LinearRegression().fit(
            A_train,
            train[feature].to_numpy(float),
        )

        train_resid[:, j] = (
            train[feature].to_numpy(float)
            - reg.predict(A_train)
        )

        test_resid[:, j] = (
            test[feature].to_numpy(float)
            - reg.predict(A_test)
        )

    feature_scaler = StandardScaler()

    X_train = feature_scaler.fit_transform(
        train_resid
    )

    X_test = feature_scaler.transform(
        test_resid
    )

    return X_train, X_test


def read_acdc_groups(acdc_root):
    rows = []

    for patient_dir in sorted(
        Path(acdc_root).glob("patient*")
    ):
        if not patient_dir.is_dir():
            continue

        info = patient_dir / "Info.cfg"

        if not info.is_file():
            continue

        text = info.read_text(
            encoding="utf-8",
            errors="replace",
        )

        match = re.search(
            r"^\s*Group\s*:\s*(.+?)\s*$",
            text,
            re.MULTILINE | re.IGNORECASE,
        )

        if match:
            rows.append({
                "patient": patient_dir.name,
                "ACDC_group": match.group(1).strip(),
            })

    return pd.DataFrame(rows)
