import pandas as pd
import numpy as np
import os

BASE = "/Users/rallabandisailesh/Desktop/aurora-github/aurora_v6"

FEAT = os.path.join(BASE, "outputs_v11/features.csv")
EF   = os.path.join(BASE, "outputs/ef_results.csv")

def load_data():
    df_f = pd.read_csv(FEAT)
    df_e = pd.read_csv(EF)
    df = pd.merge(df_f, df_e, on="Patient")
    print("Loaded:", len(df))
    return df

def dist(a, b):

    keys = ["CON_mean","EXP_mean","COH_mean",
            "CON_std","EXP_std","COH_std"]

    return sum((a[k]-b[k])**2 for k in keys)

def knn(df, row, k=10):

    dists = []

    for _, r in df.iterrows():
        if r["Patient"] == row["Patient"]:
            continue
        dists.append((dist(row, r), r))

    dists.sort(key=lambda x: x[0])
    return pd.DataFrame([r for _, r in dists[:k]])

def estimate(df):

    res = []

    for _, row in df.iterrows():

        neigh = knn(df, row, k=10)

        ef_min = neigh["EF"].min()
        ef_max = neigh["EF"].max()

        res.append({
            "Patient": row["Patient"],
            "EF_true": row["EF"],
            "EF_est": (ef_min+ef_max)/2,
            "EF_min": ef_min,
            "EF_max": ef_max,
            "Uncertainty": ef_max - ef_min
        })

    return pd.DataFrame(res)

def evaluate(df):

    df["Error"] = abs(df["EF_true"] - df["EF_est"])

    print("\n=== V11 PERFORMANCE ===")
    print("Mean Error:", round(df["Error"].mean(),4))
    print("Mean Uncertainty:", round(df["Uncertainty"].mean(),4))
    print("Leakage:", (df["EF_min"] == df["EF_true"]).sum())

if __name__ == "__main__":

    df = load_data()
    df_est = estimate(df)

    os.makedirs("outputs_v11_est", exist_ok=True)
    df_est.to_csv("outputs_v11_est/results.csv", index=False)

    evaluate(df_est)
