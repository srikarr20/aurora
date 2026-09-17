import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    GEOMETRY_FEATURES,
    compute_cke,
    load_volume,
    trajectory_geometry,
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--project-root",
    required=True,
    help="Local aurora_instrument project containing data/acdc",
)

parser.add_argument(
    "--out",
    default="analysis_output/canonical_cohort",
)

args = parser.parse_args()

root = Path(args.project_root).expanduser()
acdc = root / "data" / "acdc"

out = Path(args.out).expanduser()
out.mkdir(parents=True, exist_ok=True)

files = sorted(acdc.rglob("*_4d.nii.gz"))

rows = []
errors = []

print("===== CANONICAL C/K/E FULL-COHORT BUILD =====")
print("Files:", len(files))

for i, path in enumerate(files, 1):
    patient = path.name.replace(
        "_4d.nii.gz",
        "",
    )

    try:
        data = load_volume(path)

        C, K, E = compute_cke(data)

        geom = trajectory_geometry(
            C, K, E
        )

        H, W, Z, T = data.shape

        voxels = H * W * Z

        rows.append({
            "patient": patient,
            "H": H,
            "W": W,
            "Z": Z,
            "native_phases": T,
            "spatial_voxels": voxels,
            "log_spatial_voxels": np.log(voxels),
            **geom,
        })

        print(
            f"[{i:03d}/{len(files):03d}] "
            f"{patient}"
        )

    except Exception as exc:
        errors.append({
            "patient": patient,
            "error": f"{type(exc).__name__}: {exc}",
        })

df = pd.DataFrame(rows)

df.to_csv(
    out / "canonical_cohort_features.csv",
    index=False,
)

if errors:
    pd.DataFrame(errors).to_csv(
        out / "processing_errors.csv",
        index=False,
    )

print("\n===== ACQUISITION CORRELATIONS =====")

acquisition = [
    "native_phases",
    "Z",
    "spatial_voxels",
]

corr_rows = []

for feature in GEOMETRY_FEATURES:
    row = {"feature": feature}

    for acq in acquisition:
        row[acq] = np.corrcoef(
            df[feature],
            df[acq],
        )[0, 1]

    corr_rows.append(row)

corr = pd.DataFrame(corr_rows).set_index(
    "feature"
)

print(corr.round(4).to_string())

corr.to_csv(
    out / "acquisition_correlations.csv"
)

print("\nProcessed:", len(df))
print("Errors:", len(errors))
print("Saved:", out)
