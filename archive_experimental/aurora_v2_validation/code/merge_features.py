import pandas as pd

# paths
feat_path = "aurora_v2_validation/outputs/features.csv"
ef_path = "aurora_v2_validation/outputs/ef_results.csv"
out_path = "aurora_v2_validation/outputs/combined_features.csv"

# load
feat = pd.read_csv(feat_path)
ef = pd.read_csv(ef_path)

# merge
df = pd.merge(feat, ef, on="Patient")

# save
df.to_csv(out_path, index=False)

print("\nMerged Data:")
print(df.head())
