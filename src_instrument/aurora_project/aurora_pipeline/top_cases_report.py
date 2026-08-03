import argparse
from pathlib import Path

import pandas as pd


def tag_top_cases(csv_path, output_csv):
    csv_path = Path(csv_path)
    output_csv = Path(output_csv)

    df = pd.read_csv(csv_path)

    # Add tag column
    df["Tag"] = ""

    # Identify cases
    high_inst = df.sort_values("Instability", ascending=False).head(2)
    low_fvc = df.sort_values("FVC").head(2)
    high_fvc = df.sort_values("FVC", ascending=False).head(1)

    # Apply tags
    df.loc[high_inst.index, "Tag"] = "HIGH_INSTABILITY"
    df.loc[low_fvc.index, "Tag"] = "LOW_CONTRACTION_EXTREME"
    df.loc[high_fvc.index, "Tag"] = "HIGH_CONTRACTION"

    # Save updated CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("Tagged CSV saved:", output_csv)

    # Also print top cases
    top_cases = df[df["Tag"] != ""]
    print("\nTop Tagged Cases:\n")
    print(top_cases[["patient","FVC","Instability","Behavior","Tag"]])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    tag_top_cases(args.csv_path, args.output_csv)


if __name__ == "__main__":
    main()
