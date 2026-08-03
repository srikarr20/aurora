import argparse
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def build_final_demo(csv_path, output_dir, results_root):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    results_root = Path(results_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # LOAD DATA
    # =========================
    df = pd.read_csv(csv_path)

    # =========================
    # AUTO THRESHOLDS
    # =========================
    fvc_thresh = df["FVC"].median()
    inst_thresh = df["Instability"].quantile(0.90)

    print("Auto thresholds:")
    print("FVC threshold:", round(fvc_thresh, 3))
    print("Instability threshold:", round(inst_thresh, 3))

    # =========================
    # CLASSIFICATION (UPDATED)
    # =========================
    def classify(row):
        if row["Instability"] > inst_thresh:
            return "IRREGULAR"
        elif row["FVC"] < fvc_thresh:
            return "LOW_CONTRACTION"
        else:
            return "STABLE"

    df["Behavior_Final"] = df.apply(classify, axis=1)

    # =========================
    # SAVE UPDATED CSV
    # =========================
    final_csv = output_dir / "aurora_final_summary.csv"
    df.to_csv(final_csv, index=False)

    # =========================
    # FEATURE SPACE PLOT
    # =========================
    colors = {
        "STABLE": "green",
        "LOW_CONTRACTION": "orange",
        "IRREGULAR": "red"
    }

    plt.figure(figsize=(6,5))

    for label in df["Behavior_Final"].unique():
        sub = df[df["Behavior_Final"] == label]
        plt.scatter(sub["FVC"], sub["Instability"],
                    label=label, alpha=0.7,
                    c=colors.get(label, "gray"))

    plt.xlabel("FVC (Contraction Amplitude)")
    plt.ylabel("Instability (Temporal Variation)")
    plt.title("AURORA Final Feature Space")
    plt.legend()

    plot_path = output_dir / "feature_space_final.png"
    plt.savefig(plot_path)
    plt.close()

    # =========================
    # SELECT DEMO CASES
    # =========================
    top_inst = df.sort_values("Instability", ascending=False).head(2)
    low_fvc = df.sort_values("FVC").head(2)
    high_fvc = df.sort_values("FVC", ascending=False).head(1)

    demo_cases = pd.concat([top_inst, low_fvc, high_fvc]).drop_duplicates()

    # =========================
    # COPY FILES
    # =========================
    demo_dir = output_dir / "demo_cases"
    demo_dir.mkdir(parents=True, exist_ok=True)

    for _, row in demo_cases.iterrows():
        p = row["patient"]
        src = results_root / p
        dst = demo_dir / p

        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    # =========================
    # WRITE README
    # =========================
    readme_path = output_dir / "README.txt"

    with readme_path.open("w") as f:
        f.write("AURORA FINAL DEMO\n\n")
        f.write("System:\n")
        f.write("Segmentation-free cardiac dynamics analyzer\n\n")

        f.write("Key Idea:\n")
        f.write("Separates contraction amplitude (FVC) and temporal behavior (Instability)\n\n")

        f.write("Auto Thresholds:\n")
        f.write(f"FVC threshold: {round(fvc_thresh,3)}\n")
        f.write(f"Instability threshold: {round(inst_thresh,3)}\n\n")

        f.write("Selected Demo Cases:\n\n")

        for _, row in demo_cases.iterrows():
            f.write(f"{row['patient']}:\n")
            f.write(f"  FVC: {round(row['FVC'],3)}\n")
            f.write(f"  Instability: {round(row['Instability'],3)}\n")
            f.write(f"  Behavior: {row['Behavior_Final']}\n\n")

    print("\nFINAL DEMO PACK READY")
    print("Location:", output_dir)
    return final_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()

    build_final_demo(args.csv_path, args.output_dir, args.results_root)


if __name__ == "__main__":
    main()
