import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# AURORA FUNCTIONS (minimal)
# -----------------------------
def normalize(V):
    return np.clip((V - np.mean(V)) / (np.std(V) + 1e-8), -5, 5)

def compute_rho(V, cx=70, ct=2):
    rho = []
    for t in range(ct, V.shape[-1]):
        prev = V[..., t-ct]
        curr = V[..., t]
        mask = curr > np.percentile(curr, cx)

        if np.sum(mask) < 500:
            rho.append(0)
            continue

        diff = np.abs(curr - prev)
        val = np.mean(diff[mask]) / (np.mean(np.abs(curr[mask])) + 1e-8)
        rho.append(val)

    rho = np.array(rho)
    rho = (rho - np.min(rho)) / (np.max(rho) + 1e-8)

    # smoothing
    rho = np.convolve(rho, np.ones(3)/3, mode='same')

    return rho

def compute_metrics(rho):
    return {
        "FVC": np.max(rho) - np.min(rho),
        "Instability": np.mean(np.abs(np.diff(rho)))
    }

def compute_ef(ed_gt_path, es_gt_path):
    ed = nib.load(ed_gt_path).get_fdata()
    es = nib.load(es_gt_path).get_fdata()
    ed_area = np.sum(ed == 3)
    es_area = np.sum(es == 3)
    return (ed_area - es_area) / (ed_area + 1e-8)

def run_full_pipeline(data_root, output_dir):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for patient_path in sorted(data_root.iterdir()):
        if not patient_path.is_dir():
            continue

        try:
            files = [path.name for path in patient_path.iterdir()]

            vol = [f for f in files if "_4d.nii.gz" in f][0]
            ed = [f for f in files if "frame01_gt" in f][0]
            es = [f for f in files if "frame12_gt" in f][0]

            vol_path = patient_path / vol
            ed_path = patient_path / ed
            es_path = patient_path / es

            V = nib.load(vol_path).get_fdata()
            V = normalize(V)

            rho = compute_rho(V)
            m = compute_metrics(rho)
            ef = compute_ef(ed_path, es_path)

            row = {
                "Patient": patient_path.name,
                "EF": ef,
                **m
            }

            rows.append(row)
            print(f"{patient_path.name} done")

        except Exception as e:
            print(f"Skipping {patient_path.name}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "dataset.csv", index=False)

    X = df[['FVC', 'Instability']]
    y = df['EF']

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X, y)

    df['EF_pred_ridge'] = ridge.predict(X)
    df['EF_pred_rf'] = rf.predict(X)

    df.to_csv(output_dir / "results_with_predictions.csv", index=False)

    corr = np.corrcoef(df['EF'], df['FVC'])[0, 1]

    with (output_dir / "summary.txt").open("w") as f:
        f.write(f"Correlation EF-FVC: {corr}\n")

    print("\nFULL PIPELINE COMPLETE")
    print("Output:", output_dir)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    run_full_pipeline(args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
