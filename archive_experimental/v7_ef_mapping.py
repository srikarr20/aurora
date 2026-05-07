import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# LOAD DATA
# =========================
def load_data(aci_path, ef_path):

    df_aci = pd.read_csv(aci_path)
    df_ef = pd.read_csv(ef_path)

    # merge on Patient
    df = pd.merge(df_aci, df_ef, on="Patient")

    print("\nMerged Data:")
    print(df.head())

    return df


# =========================
# REGIME-BASED ANALYSIS
# =========================
def analyze_by_regime(df):

    print("\n=== REGIME ANALYSIS ===")

    for label in ["LOW_CONTRACTION", "NORMAL", "IRREGULAR"]:

        subset = df[df["Label"] == label]

        if len(subset) == 0:
            continue

        print(f"\n--- {label} ---")
        print(f"Count: {len(subset)}")

        print("EF range:",
              round(subset["EF"].min(), 3),
              "→",
              round(subset["EF"].max(), 3))

        print("Mean EF:",
              round(subset["EF"].mean(), 3))


# =========================
# PLOT ACI vs EF
# =========================
def plot_aci_vs_ef(df, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    plt.figure()

    for label in ["LOW_CONTRACTION", "NORMAL", "IRREGULAR"]:
        subset = df[df["Label"] == label]
        plt.scatter(subset["ACI"], subset["EF"], label=label)

    plt.xlabel("ACI (Spatial Coherence)")
    plt.ylabel("EF")
    plt.title("ACI vs EF (Regime-Aware)")
    plt.legend()

    path = os.path.join(out_dir, "aci_vs_ef.png")
    plt.savefig(path)

    print(f"\nSaved: {path}")


# =========================
# PLOT TCI vs EF
# =========================
def plot_tci_vs_ef(df, out_dir):

    plt.figure()

    for label in ["LOW_CONTRACTION", "NORMAL", "IRREGULAR"]:
        subset = df[df["Label"] == label]
        plt.scatter(subset["TCI"], subset["EF"], label=label)

    plt.xlabel("TCI (Temporal Variability)")
    plt.ylabel("EF")
    plt.title("TCI vs EF (Regime-Aware)")
    plt.legend()

    path = os.path.join(out_dir, "tci_vs_ef.png")
    plt.savefig(path)

    print(f"Saved: {path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    aci_path = "outputs/aci_results.csv"
    ef_path = "outputs/ef_results.csv"   # YOU MUST HAVE THIS

    out_dir = "outputs/ef_analysis"

    df = load_data(aci_path, ef_path)

    analyze_by_regime(df)

    plot_aci_vs_ef(df, out_dir)
    plot_tci_vs_ef(df, out_dir)
