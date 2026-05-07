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

    print("\nLoaded:", len(df), "patients")
    return df


# =========================
# KNN-BASED LOCAL MEASUREMENT
# =========================
def get_neighbors(df, row, k=10):

    df_temp = df.copy()

    # compute distance in observable space
    df_temp["dist"] = (
        (df_temp["ACI"] - row["ACI"])**2 +
        (df_temp["TCI"] - row["TCI"])**2
    )

    # remove self → prevents leakage
    df_temp = df_temp[df_temp["Patient"] != row["Patient"]]

    # get nearest neighbors
    neighbors = df_temp.nsmallest(k, "dist")

    return neighbors


# =========================
# ESTIMATE EF WITH LOCAL CONSTRAINTS
# =========================
def estimate_ef(df, k=10):

    estimates = []

    for _, row in df.iterrows():

        neighbors = get_neighbors(df, row, k=k)

        if len(neighbors) < 3:
            continue

        ef_min = neighbors["EF"].min()
        ef_max = neighbors["EF"].max()

        ef_est = (ef_min + ef_max) / 2
        uncertainty = ef_max - ef_min

        estimates.append({
            "Patient": row["Patient"],
            "ACI": row["ACI"],
            "TCI": row["TCI"],
            "EF_true": row["EF"],
            "EF_est": ef_est,
            "EF_min": ef_min,
            "EF_max": ef_max,
            "Uncertainty": uncertainty,
            "Neighbor_count": len(neighbors)
        })

    return pd.DataFrame(estimates)


# =========================
# EVALUATION
# =========================
def evaluate(df_est):

    df_est["Error"] = np.abs(df_est["EF_true"] - df_est["EF_est"])

    print("\n=== PERFORMANCE ===")
    print("Mean Error:", round(df_est["Error"].mean(), 4))
    print("Mean Uncertainty:", round(df_est["Uncertainty"].mean(), 4))

    # check leakage condition
    exact_matches = (df_est["EF_min"] == df_est["EF_true"]).sum()
    print("EF_min == EF_true count:", exact_matches)

    return df_est


# =========================
# SAVE
# =========================
def save(df_est):

    os.makedirs("outputs/ef_estimation_v74", exist_ok=True)

    path = "outputs/ef_estimation_v74/ef_estimates.csv"
    df_est.to_csv(path, index=False)

    print("\nSaved:", path)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    df = load_data()

    df_est = estimate_ef(df, k=10)

    df_est = evaluate(df_est)

    save(df_est)
