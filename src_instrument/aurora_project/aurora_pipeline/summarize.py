import argparse
import json
from pathlib import Path


def summarize(results_root):
    root = Path(results_root)
    total = 0
    low_signal = 0

    for patient_dir in root.iterdir():
        path = patient_dir / "report.json"
        if not path.exists():
            continue

        with path.open() as f:
            r = json.load(f)

        total += 1
        if r.get("quality_flag") == "LOW_SIGNAL":
            low_signal += 1

    print("Total:", total)
    print("Low signal:", low_signal)
    return total, low_signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()

    summarize(args.results_root)


if __name__ == "__main__":
    main()
