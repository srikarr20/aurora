import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from common import read_acdc_groups


parser = argparse.ArgumentParser()

parser.add_argument(
    "--project-root",
    required=True,
)

parser.add_argument(
    "--assignments",
    required=True,
)

parser.add_argument(
    "--out",
    default="analysis_output/clinical_reveal",
)

args = parser.parse_args()

root = Path(args.project_root).expanduser()
out = Path(args.out).expanduser()
out.mkdir(parents=True, exist_ok=True)

assign = pd.read_csv(
    Path(args.assignments).expanduser()
)

cluster_col = "residualized_geometry_k2"

groups = read_acdc_groups(
    root / "data" / "acdc"
)

df = assign[
    ["patient", cluster_col]
].merge(
    groups,
    on="patient",
    validate="one_to_one",
)

table = pd.crosstab(
    df["ACDC_group"],
    df[cluster_col],
)

row_pct = table.div(
    table.sum(axis=1),
    axis=0,
) * 100

chi2, p, dof, expected = chi2_contingency(
    table.to_numpy()
)

n = table.to_numpy().sum()

v = np.sqrt(
    (chi2 / n)
    /
    min(
        table.shape[0] - 1,
        table.shape[1] - 1,
    )
)

print("===== ACDC GROUP × FROZEN CLUSTER =====")
print(table.to_string())

print("\n===== ROW PERCENTAGES =====")
print(row_pct.round(1).to_string())

print("\n===== ASSOCIATION =====")
print("N:", n)
print("Chi-square:", chi2)
print("p-value:", p)
print("Cramer's V:", v)

table.to_csv(
    out / "acdc_group_by_cluster_counts.csv"
)

row_pct.to_csv(
    out / "acdc_group_by_cluster_percent.csv"
)

df.to_csv(
    out / "clusters_with_acdc_groups.csv",
    index=False,
)
