import pandas as pd

feat = pd.read_csv("aurora_v2_validation/outputs/spatial_features.csv")
ef = pd.read_csv("aurora_v2_validation/outputs/ef_results.csv")

df = pd.merge(feat, ef, on="Patient")

df.to_csv("aurora_v2_validation/outputs/combined_spatial.csv", index=False)

print(df.head())
