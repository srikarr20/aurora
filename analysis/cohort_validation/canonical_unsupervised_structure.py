import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from common import (
    ACQUISITION_FEATURES,
    GEOMETRY_FEATURES,
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--features",
    required=True,
)

parser.add_argument(
    "--out",
    default="analysis_output/unsupervised_structure",
)

args = parser.parse_args()

df = pd.read_csv(
    Path(args.features).expanduser()
).dropna(
    subset=GEOMETRY_FEATURES
    + ACQUISITION_FEATURES
).reset_index(drop=True)

out = Path(args.out).expanduser()
out.mkdir(parents=True, exist_ok=True)

A = StandardScaler().fit_transform(
    df[ACQUISITION_FEATURES].to_numpy(float)
)

residuals = np.zeros(
    (len(df), len(GEOMETRY_FEATURES))
)

for j, feature in enumerate(
    GEOMETRY_FEATURES
):
    y = df[feature].to_numpy(float)

    reg = LinearRegression().fit(
        A,
        y,
    )

    residuals[:, j] = (
        y - reg.predict(A)
    )

X = StandardScaler().fit_transform(
    residuals
)

rng = np.random.default_rng(20260916)

rows = []
assignments = df[["patient"]].copy()

print("===== CANONICAL UNSUPERVISED STRUCTURE =====")
print("Subjects:", len(df))
print("Diagnosis labels used: NO")

for k in range(2, 7):
    reference = KMeans(
        n_clusters=k,
        n_init=100,
        random_state=20260916,
    ).fit_predict(X)

    assignments[
        f"residualized_geometry_k{k}"
    ] = reference

    silhouette = silhouette_score(
        X,
        reference,
    )

    ch = calinski_harabasz_score(
        X,
        reference,
    )

    db = davies_bouldin_score(
        X,
        reference,
    )

    counts = np.bincount(
        reference,
        minlength=k,
    )

    seed_ari = []

    for seed in range(50):
        labels = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=seed,
        ).fit_predict(X)

        seed_ari.append(
            adjusted_rand_score(
                reference,
                labels,
            )
        )

    subsample_ari = []

    n = len(X)
    sample_n = int(round(0.80 * n))

    for iteration in range(100):
        idx = np.sort(
            rng.choice(
                n,
                size=sample_n,
                replace=False,
            )
        )

        labels = KMeans(
            n_clusters=k,
            n_init=50,
            random_state=10000 + iteration,
        ).fit_predict(X[idx])

        subsample_ari.append(
            adjusted_rand_score(
                reference[idx],
                labels,
            )
        )

    rows.append({
        "k": k,
        "silhouette": silhouette,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "seed_ARI_mean": np.mean(seed_ari),
        "subsample_ARI_mean": np.mean(
            subsample_ari
        ),
        "subsample_ARI_p10": np.percentile(
            subsample_ari,
            10,
        ),
    })

    print(
        f"k={k} "
        f"sil={silhouette:.3f} "
        f"CH={ch:.1f} "
        f"DB={db:.3f} "
        f"clusters={list(counts)} "
        f"subARI={np.mean(subsample_ari):.3f}"
    )

pd.DataFrame(rows).to_csv(
    out / "cluster_model_comparison.csv",
    index=False,
)

assignments.to_csv(
    out / "cluster_assignments.csv",
    index=False,
)

print("\nSaved:", out)
