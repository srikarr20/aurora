import argparse
import json
from pathlib import Path

import pandas as pd


def build_master(results_root):
    root = Path(results_root)
    rows = []

    for patient_dir in root.iterdir():
        path = patient_dir / "report.json"
        if not path.exists():
            continue

        with path.open() as f:
            r = json.load(f)

        row = {
            "patient": patient_dir.name,
            "FVC": r.get("FVC", 0),
            "Instability": r.get("instability_index", 0),
            "Coherence": r.get("coherence", 0),
            "Variability": r.get("variability", 0),
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = root / "master.csv"
    df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()

    build_master(args.results_root)


if __name__ == "__main__":
    main()
