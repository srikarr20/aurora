import pandas as pd

feat = pd.read_csv("aurora_v2_validation/outputs/v5_features.csv")
ef = pd.read_csv("aurora_v2_validation/outputs/ef_results.csv")

df = pd.merge(feat, ef, on="Patient")

df.to_csv("aurora_v2_validation/outputs/v5_combined.csv", index=False)

print(df.head())
