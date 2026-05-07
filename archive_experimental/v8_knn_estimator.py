import pandas as pd
import numpy as np
import os

BASE = "/Users/rallabandisailesh/Desktop/aurora-github/aurora_v6"

FEATURE_PATH = os.path.join(BASE, "outputs_v8/aci_tci_regional.csv")
EF_PATH      = os.path.join(BASE, "outputs/ef_results.csv")
SAVE_DIR     = os.path.join(BASE, "outputs_v8_est")


# =========================
# LOAD DATA
# =========================
def load_data():

    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError("Feature file missing")

    df_feat = pd.read_csv(FEATURE_PATH)

    if df_feat.empty:
        raise ValueError("Feature file is EMPTY")

    df_ef = pd.read_csv(EF_PATH)

    df = pd.merge(df_feat, df_ef, on="Patient")

    print("Loaded:", len(df))
    return df


# =========================
# DISTANCE
# =========================
def compute_distance(r1, r2):

    d = 0
    for i in range(4):
        d += (r1[f"ACI_{i}"] - r2[f"ACI_{i}"])**2
        d += (r1[f"TCI_{i}"] - r2[f"TCI_{i}"])**2
    return d


# =========================
# KNN
# =========================
def get_neighbors(df, row, k=10):

    distances = []

    for _, r in df.iterrows():

        if r["Patient"] == row["Patient"]:
            continue

        distances.append((compute_distance(row, r), r))

    distances.sort(key=lambda x: x[0])

    return pd.DataFrame([r for _, r in distances[:k]])


# =========================
# ESTIMATE
# =========================
def estimate(df, k=10):

    results = []

    for _, row in df.iterrows():

        neighbors = get_neighbors(df, row, k)

        if len(neighbors) < 3:
            continue

        ef_min = neighbors["EF"].min()
        ef_max = neighbors["EF"].max()

        ef_est = (ef_min + ef_max) / 2
        uncertainty = ef_max - ef_min

        results.append({
            "Patient": row["Patient"],
            "EF_true": row["EF"],
            "EF_est": ef_est,
            "EF_min": ef_min,
            "EF_max": ef_max,
            "Uncertainty": uncertainty
        })

    return pd.DataFrame(results)


# =========================
# EVALUATE
# =========================
def evaluate(df):

    df["Error"] = np.abs(df["EF_true"] - df["EF_est"])

    print("\n=== PERFORMANCE ===")
    print("Mean Error:", round(df["Error"].mean(), 4))
    print("Mean Uncertainty:", round(df["Uncertainty"].mean(), 4))

    print("Leakage:", (df["EF_min"] == df["EF_true"]).sum())

    corr = df["Uncertainty"].corr(df["Error"])
    print("Error vs Uncertainty corr:", round(corr, 4))


# =========================
# SAVE
# =========================
def save(df):

    os.makedirs(SAVE_DIR, exist_ok=True)

    path = os.path.join(SAVE_DIR, "results.csv")
    df.to_csv(path, index=False)

    print("Saved:", path)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    df = load_data()

    df_est = estimate(df)

    save(df_est)

    evaluate(df_est)
