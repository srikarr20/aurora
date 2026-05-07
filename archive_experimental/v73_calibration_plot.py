import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# LOAD ESTIMATES
# =========================
def load_data():

    df = pd.read_csv("outputs/ef_estimation/ef_estimates.csv")

    print("\nLoaded:", len(df), "rows")
    print(df.head())

    return df


# =========================
# PLOT ERROR vs UNCERTAINTY
# =========================
def plot_error_vs_uncertainty(df):

    plt.figure()

    plt.scatter(df["Uncertainty"], df["Error"])

    plt.xlabel("Uncertainty (EF_max - EF_min)")
    plt.ylabel("Absolute Error |EF_true - EF_est|")
    plt.title("Calibration: Error vs Uncertainty")

    # ideal line (error = uncertainty)
    max_val = max(df["Uncertainty"].max(), df["Error"].max())
    plt.plot([0, max_val], [0, max_val], linestyle="--")

    os.makedirs("outputs/ef_estimation", exist_ok=True)
    path = "outputs/ef_estimation/error_vs_uncertainty.png"
    plt.savefig(path)

    print("\nSaved:", path)


# =========================
# BINNED ANALYSIS
# =========================
def binned_analysis(df):

    df["bin"] = pd.qcut(df["Uncertainty"], 4, duplicates='drop')

    print("\n=== BINNED CALIBRATION ===")

    for b in df["bin"].unique():

        subset = df[df["bin"] == b]

        print(f"\nBin: {b}")
        print("Mean Uncertainty:", round(subset["Uncertainty"].mean(), 3))
        print("Mean Error:", round(subset["Error"].mean(), 3))
        print("Count:", len(subset))


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    df = load_data()

    plot_error_vs_uncertainty(df)

    binned_analysis(df)
