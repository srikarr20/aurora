import pandas as pd

df = pd.read_csv("aurora_v2_validation/outputs/v5_combined.csv")

# normalize features
df["E_n"] = df["ENERGY"] / df["ENERGY"].max()
df["A_n"] = df["ACTIVE_AREA"] / df["ACTIVE_AREA"].max()
df["S_n"] = df["SPATIAL_STD"] / df["SPATIAL_STD"].max()

# AMI (weighted)
df["AMI"] = 0.4 * df["A_n"] + 0.4 * df["S_n"] + 0.2 * df["E_n"]

# save
df.to_csv("aurora_v2_validation/outputs/ami_results.csv", index=False)

print(df[["Patient","AMI","EF"]].head())
