import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# LOAD MERGED DATA
# =========================
def load_data():

    df_aci = pd.read_csv("outputs/aci_results.csv")
    df_ef  = pd.read_csv("outputs/ef_results.csv")

    df = pd.merge(df_aci, df_ef, on="Patient")

    print("\nLoaded:", len(df), "patients")
    return df


# =========================
# BINNING SPACE (ACI, TCI)
# =========================
def compute_feasibility(df):

    # define bins
    aci_bins = np.linspace(df["ACI"].min(), df["ACI"].max(), 8)
    tci_bins = np.linspace(df["TCI"].min(), df["TCI"].max(), 8)

    results = []

    for i in range(len(aci_bins)-1):
        for j in range(len(tci_bins)-1):

            subset = df[
                (df["ACI"] >= aci_bins[i]) &
                (df["ACI"] <  aci_bins[i+1]) &
                (df["TCI"] >= tci_bins[j]) &
                (df["TCI"] <  tci_bins[j+1])
            ]

            if len(subset) < 3:
                continue

            ef_min = subset["EF"].min()
            ef_max = subset["EF"].max()

            results.append({
                "ACI_mid": (aci_bins[i] + aci_bins[i+1]) / 2,
                "TCI_mid": (tci_bins[j] + tci_bins[j+1]) / 2,
                "EF_min": ef_min,
                "EF_max": ef_max,
                "count": len(subset)
            })

    return pd.DataFrame(results)


# =========================
# VISUALIZE FEASIBLE REGION
# =========================
def plot_feasibility(df, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    plt.figure()

    # draw vertical EF ranges
    for _, row in df.iterrows():
        plt.plot(
            [row["ACI_mid"], row["ACI_mid"]],
            [row["EF_min"], row["EF_max"]],
            linewidth=2
        )

    plt.xlabel("ACI")
    plt.ylabel("EF range")
    plt.title("EF Feasibility vs ACI")

    path = os.path.join(out_dir, "ef_feasibility_aci.png")
    plt.savefig(path)

    print("Saved:", path)


# =========================
# TCI FEASIBILITY
# =========================
def plot_feasibility_tci(df, out_dir):

    plt.figure()

    for _, row in df.iterrows():
        plt.plot(
            [row["TCI_mid"], row["TCI_mid"]],
            [row["EF_min"], row["EF_max"]],
            linewidth=2
        )

    plt.xlabel("TCI")
    plt.ylabel("EF range")
    plt.title("EF Feasibility vs TCI")

    path = os.path.join(out_dir, "ef_feasibility_tci.png")
    plt.savefig(path)

    print("Saved:", path)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    df = load_data()

    feasibility = compute_feasibility(df)

    print("\nFeasibility samples:")
    print(feasibility.head())

    out_dir = "outputs/ef_feasibility"

    plot_feasibility(feasibility, out_dir)
    plot_feasibility_tci(feasibility, out_dir)
