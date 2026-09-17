import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import joblib


# =========================
# LOAD VALIDATION DATA
# =========================
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


# =========================
# TRAIN MODELS
# =========================
def train_models(df, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # features
    X = df[['FVC', 'Instability']]

    # target
    y = df['EF']

    # ------------------------
    # RIDGE (main model)
    # ------------------------
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)

    # ------------------------
    # RANDOM FOREST
    # ------------------------
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X, y)

    # ------------------------
    # CLUSTERING (phenotypes)
    # ------------------------
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)

    df['cluster'] = clusters

    # save models
    joblib.dump(ridge, model_dir / "ridge.pkl")
    joblib.dump(rf, model_dir / "rf.pkl")
    joblib.dump(kmeans, model_dir / "kmeans.pkl")

    # save cluster data
    df.to_csv(model_dir / "clustered_data.csv", index=False)

    print("Models trained and saved")

    return ridge, rf, kmeans


# =========================
# PREDICT (AURORA OUTPUT)
# =========================
def predict(FVC, instability, model_dir):
    model_dir = Path(model_dir)

    ridge = joblib.load(model_dir / "ridge.pkl")
    kmeans = joblib.load(model_dir / "kmeans.pkl")

    X = np.array([[FVC, instability]])

    ef_pred = ridge.predict(X)[0]
    cluster = kmeans.predict(X)[0]

    # risk mapping
    if ef_pred < 0.4:
        risk = "LOW FUNCTION"
    elif ef_pred < 0.6:
        risk = "MEDIUM"
    else:
        risk = "NORMAL"

    return {
        "predicted_EF": float(ef_pred),
        "cluster": int(cluster),
        "risk": risk
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    df = load_data(args.csv_path)
    train_models(df, args.model_dir)


if __name__ == "__main__":
    main()
