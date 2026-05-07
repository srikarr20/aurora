import pandas as pd
import numpy as np
import os

# =========================
# LOAD DATA
# =========================
def load_data():

    df_aci = pd.read_csv("outputs/aci_results.csv")
    df_ef  = pd.read_csv("outputs/ef_results.csv")

    df = pd.merge(df_aci, df_ef, on="Patient")

    return df


# =========================
# BUILD FEASIBILITY TABLE
# =========================
def build_feasibility(df):

    aci_bins = np.linspace(df["ACI"].min(), df["ACI"].max(), 8)
    tci_bins = np.linspace(df["TCI"].min(), df["TCI"].max(), 8)

    table = []

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

            table.append({
                "aci_min": aci_bins[i],
                "aci_max": aci_bins[i+1],
                "tci_min": tci_bins[j],
                "tci_max": tci_bins[j+1],
                "EF_min": subset["EF"].min(),
                "EF_max": subset["EF"].max()
            })

    return pd.DataFrame(table)


# =========================
# ESTIMATE EF FOR EACH PATIENT
# =========================
def estimate_ef(df, table):

    estimates = []

    for _, row in df.iterrows():

        aci = row["ACI"]
        tci = row["TCI"]

        match = table[
            (table["aci_min"] <= aci) & (aci < table["aci_max"]) &
            (table["tci_min"] <= tci) & (tci < table["tci_max"])
        ]

        if len(match) == 0:
            continue

        ef_min = match.iloc[0]["EF_min"]
        ef_max = match.iloc[0]["EF_max"]

        ef_est = (ef_min + ef_max) / 2
        uncertainty = ef_max - ef_min

        estimates.append({
            "Patient": row["Patient"],
            "ACI": aci,
            "TCI": tci,
            "EF_true": row["EF"],
            "EF_est": ef_est,
            "EF_min": ef_min,
            "EF_max": ef_max,
            "Uncertainty": uncertainty
        })

    return pd.DataFrame(estimates)


# =========================
# EVALUATE PERFORMANCE
# =========================
def evaluate(df_est):

    df_est["Error"] = np.abs(df_est["EF_true"] - df_est["EF_est"])

    print("\n=== PERFORMANCE ===")
    print("Mean Error:", round(df_est["Error"].mean(), 4))
    print("Mean Uncertainty:", round(df_est["Uncertainty"].mean(), 4))

    print("\nSample:")
    print(df_est.head())


# =========================
# SAVE RESULTS
# =========================
def save_results(df_est):

    os.makedirs("outputs/ef_estimation", exist_ok=True)

    path = "outputs/ef_estimation/ef_estimates.csv"
    df_est.to_csv(path, index=False)

    print("\nSaved:", path)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    df = load_data()

    table = build_feasibility(df)

    df_est = estimate_ef(df, table)

    evaluate(df_est)

    save_results(df_est)
