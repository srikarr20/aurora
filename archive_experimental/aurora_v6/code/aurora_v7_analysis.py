import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# LOAD RESULTS
# =========================
def load_results(csv_path):
    df = pd.read_csv(csv_path)
    print("\nLoaded Data:")
    print(df.head())
    return df


# =========================
# REFINED CLASSIFICATION
# =========================
def refine_labels(df):

    labels = []

    for _, row in df.iterrows():

        aci = row["ACI"]
        tci = row["TCI"]

        if aci < 0.045:
            label = "LOW_CONTRACTION"

        elif tci > 0.02:
            label = "IRREGULAR"

        else:
            label = "NORMAL"

        labels.append(label)

    df["Refined_Label"] = labels
    return df


# =========================
# PLOT ACI vs TCI
# =========================
def plot_distribution(df, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    plt.figure()

    for label in ["LOW_CONTRACTION", "NORMAL", "IRREGULAR"]:
        subset = df[df["Refined_Label"] == label]
        plt.scatter(subset["ACI"], subset["TCI"], label=label)

    plt.xlabel("ACI (Spatial Coherence)")
    plt.ylabel("TCI (Temporal Variability)")
    plt.title("AURORA V7: ACI vs TCI Distribution")
    plt.legend()

    save_path = os.path.join(out_dir, "aci_tci_plot.png")
    plt.savefig(save_path)

    print(f"\nSaved plot: {save_path}")


# =========================
# SUMMARY
# =========================
def print_summary(df):

    print("\n=== CLASS DISTRIBUTION ===")
    print(df["Refined_Label"].value_counts())

    print("\n=== ACI RANGE ===")
    print(df["ACI"].min(), "→", df["ACI"].max())

    print("\n=== TCI RANGE ===")
    print(df["TCI"].min(), "→", df["TCI"].max())


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    csv_path = "outputs/aci_results.csv"
    out_dir = "outputs/plots"

    df = load_results(csv_path)
    df = refine_labels(df)

    print_summary(df)
    plot_distribution(df, out_dir)
