# =========================================
# AURORA PoC
# =========================================

import os
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def interpret(label):
    if label == "STABLE":
        return "Consistent contraction pattern"
    elif label == "IRREGULAR":
        return "Temporal inconsistency detected"
    elif label == "LOW_CONTRACTION":
        return "Reduced contraction amplitude signal"
    else:
        return "Unknown"


def run_poc(data_root, output_root):
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []

    for patient_dir in sorted(data_root.iterdir()):
        if not patient_dir.is_dir():
            continue

        try:
            vol_path = [path for path in patient_dir.iterdir() if "_4d.nii.gz" in path.name][0]
            out_dir = output_root / patient_dir.name

            cmd = [
                sys.executable,
                "-m",
                "src_instrument.aurora_project.aurora_pipeline.aurora_run",
                "--input",
                str(vol_path),
                "--output",
                str(out_dir),
            ]
            print(f"Running {patient_dir.name}")
            subprocess.run(cmd, check=False)

            report_path = out_dir / "report.json"

            if not report_path.exists():
                continue

            with report_path.open() as f:
                r = json.load(f)

            label = r.get("behavior_label", "UNKNOWN")

            rows.append({
                "patient": patient_dir.name,
                "FVC": r.get("FVC", 0),
                "Instability": r.get("instability_index", 0),
                "Coherence": r.get("coherence", 0),
                "Variability": r.get("variability", 0),
                "Quality": r.get("quality_flag", "UNKNOWN"),
                "Behavior": label,
                "Interpretation": interpret(label),
            })

        except Exception as e:
            print(f"Skipping {patient_dir.name}: {e}")

    df = pd.DataFrame(rows)

    summary_path = output_root / "aurora_poc_summary.csv"
    df.to_csv(summary_path, index=False)

    print("\nAURORA PoC COMPLETE")
    print("Saved:", summary_path)
    print("\nBehavior distribution:")
    print(df["Behavior"].value_counts())
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    run_poc(args.data_root, args.output_root)


if __name__ == "__main__":
    main()
