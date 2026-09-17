import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans

from common import (
    GEOMETRY_FEATURES,
    read_acdc_groups,
    residualize_train_test,
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--project-root",
    required=True,
)

parser.add_argument(
    "--features",
    required=True,
)

parser.add_argument(
    "--out",
    default="analysis_output/heldout_permutation",
)

parser.add_argument(
    "--splits",
    type=int,
    default=200,
)

parser.add_argument(
    "--permutations",
    type=int,
    default=1000,
)

args = parser.parse_args()

root = Path(args.project_root).expanduser()

out = Path(args.out).expanduser()
out.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(
    Path(args.features).expanduser()
)

groups = read_acdc_groups(
    root / "data" / "acdc"
)

df = features.merge(
    groups,
    on="patient",
    validate="one_to_one",
).reset_index(drop=True)

rng = np.random.default_rng(20260916)

all_groups = sorted(
    df["ACDC_group"].unique()
)


def cramers_v(table):
    try:
        chi2, _, _, _ = chi2_contingency(
            np.asarray(table)
        )
    except ValueError:
        return np.nan

    arr = np.asarray(table)

    n = arr.sum()

    denom = min(
        arr.shape[0] - 1,
        arr.shape[1] - 1,
    )

    if n == 0 or denom <= 0:
        return np.nan

    return np.sqrt(
        (chi2 / n) / denom
    )


split_records = []
observed_v = []

n = len(df)
test_n = int(round(0.30 * n))

print("===== HELD-OUT REPLICATION =====")
print("Subjects:", n)
print("Splits:", args.splits)
print("Diagnosis used during fitting: NO")

for rep in range(args.splits):
    test_idx = np.sort(
        rng.choice(
            n,
            size=test_n,
            replace=False,
        )
    )

    train_mask = np.ones(
        n,
        dtype=bool,
    )

    train_mask[test_idx] = False

    train = df.iloc[
        np.flatnonzero(train_mask)
    ]

    test = df.iloc[test_idx]

    X_train, X_test = (
        residualize_train_test(
            train,
            test,
        )
    )

    km = KMeans(
        n_clusters=2,
        n_init=100,
        random_state=100000 + rep,
    )

    km.fit(X_train)

    labels = km.predict(X_test)

    turn_idx = GEOMETRY_FEATURES.index(
        "mean_turn_angle_rad"
    )

    sync_idx = GEOMETRY_FEATURES.index(
        "mean_abs_observable_sync"
    )

    organization_score = (
        km.cluster_centers_[:, sync_idx]
        - km.cluster_centers_[:, turn_idx]
    )

    s_cluster = int(
        np.argmax(organization_score)
    )

    modes = np.where(
        labels == s_cluster,
        "S",
        "A",
    )

    split_records.append(
        (test_idx, modes)
    )

    tmp = pd.DataFrame({
        "group": df.iloc[
            test_idx
        ]["ACDC_group"].to_numpy(),
        "mode": modes,
    })

    table = pd.crosstab(
        tmp["group"],
        tmp["mode"],
    ).reindex(
        index=all_groups,
        columns=["A", "S"],
        fill_value=0,
    )

    observed_v.append(
        cramers_v(table)
    )

observed_v = np.asarray(
    observed_v,
    dtype=float,
)

observed_stat = np.nanmedian(
    observed_v
)

print(
    "Observed median held-out V:",
    f"{observed_stat:.4f}",
)

true_labels = df[
    "ACDC_group"
].to_numpy()

null_stats = []

for _ in range(args.permutations):
    shuffled = rng.permutation(
        true_labels
    )

    values = []

    for test_idx, modes in split_records:
        tmp = pd.DataFrame({
            "group": shuffled[test_idx],
            "mode": modes,
        })

        table = pd.crosstab(
            tmp["group"],
            tmp["mode"],
        ).reindex(
            index=all_groups,
            columns=["A", "S"],
            fill_value=0,
        )

        values.append(
            cramers_v(table)
        )

    null_stats.append(
        np.nanmedian(values)
    )

null_stats = np.asarray(
    null_stats
)

empirical_p = (
    1
    + np.sum(
        null_stats >= observed_stat
    )
) / (
    args.permutations + 1
)

print("\n===== PERMUTATION NULL =====")
print(
    "Null median V mean:",
    f"{null_stats.mean():.4f}",
)
print(
    "Null median V 95th:",
    f"{np.percentile(null_stats,95):.4f}",
)
print(
    "Null median V 99th:",
    f"{np.percentile(null_stats,99):.4f}",
)
print(
    "Observed median V:",
    f"{observed_stat:.4f}",
)
print(
    "Empirical p-value:",
    f"{empirical_p:.6f}",
)

pd.DataFrame({
    "heldout_cramers_v": observed_v
}).to_csv(
    out / "observed_heldout_distribution.csv",
    index=False,
)

pd.DataFrame({
    "null_median_cramers_v": null_stats
}).to_csv(
    out / "permutation_null_distribution.csv",
    index=False,
)
