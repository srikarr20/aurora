import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def classify_master(results_root):
    root = Path(results_root)

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv(root / "master.csv")

    # -----------------------------
    # DATA-DRIVEN THRESHOLDS
    # -----------------------------
    fvc_q25 = df["FVC"].quantile(0.25)
    fvc_q50 = df["FVC"].quantile(0.50)
    fvc_q75 = df["FVC"].quantile(0.75)

    inst_q50 = df["Instability"].quantile(0.50)
    inst_q75 = df["Instability"].quantile(0.75)

    print("Thresholds:")
    print(f"FVC Q25: {fvc_q25:.3f}")
    print(f"FVC Q50: {fvc_q50:.3f}")
    print(f"FVC Q75: {fvc_q75:.3f}")
    print(f"Instability Q50: {inst_q50:.3f}")
    print(f"Instability Q75: {inst_q75:.3f}")

    # -----------------------------
    # CLASSIFIER
    # -----------------------------
    def label(row):
        if row["FVC"] < fvc_q25:
            return "LOW_CONTRACTION"

        elif row["Instability"] > inst_q75:
            return "IRREGULAR"

        else:
            return "STABLE"

    df["label"] = df.apply(label, axis=1)

    # -----------------------------
    # SAVE LABELED DATA
    # -----------------------------
    out_csv = root / "master_labeled.csv"
    df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)

    # -----------------------------
    # COUNT DISTRIBUTION
    # -----------------------------
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    # -----------------------------
    # VISUALIZE
    # -----------------------------
    colors = {
        "STABLE": "green",
        "LOW_CONTRACTION": "red",
        "IRREGULAR": "orange"
    }

    plt.figure(figsize=(6,6))

    for label_name, group in df.groupby("label"):
        plt.scatter(group["FVC"], group["Instability"],
                    c=colors[label_name], label=label_name)

    plt.xlabel("FVC")
    plt.ylabel("Instability")
    plt.title("AURORA Labeled Feature Space")
    plt.legend()

    plot_path = root / "labeled_space.png"
    plt.savefig(plot_path)
    plt.close()

    print("Saved:", plot_path)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()

    classify_master(args.results_root)


if __name__ == "__main__":
    main()
