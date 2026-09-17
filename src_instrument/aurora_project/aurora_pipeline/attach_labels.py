import argparse
import json
from pathlib import Path

import pandas as pd


def attach_labels(results_root):
    root = Path(results_root)
    df = pd.read_csv(root / "master_labeled.csv")

    for _, row in df.iterrows():

        p = row["patient"]
        label = row["label"]

        report_path = root / p / "report.json"

        if not report_path.exists():
            continue

        with report_path.open() as f:
            r = json.load(f)

        r["behavior_label"] = label

        with report_path.open("w") as f:
            json.dump(r, f, indent=4)

    print("Labels attached to reports")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()

    attach_labels(args.results_root)


if __name__ == "__main__":
    main()
