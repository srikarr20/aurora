import pandas as pd

# paths
msi_path = "aurora_v2_validation/outputs/final_batch/results.csv"
ef_path = "aurora_v2_validation/outputs/ef_results.csv"
out_path = "aurora_v2_validation/outputs/combined.csv"

# load data
msi_df = pd.read_csv(msi_path)
ef_df = pd.read_csv(ef_path)

# standardize column names
msi_df.columns = ["Patient", "MSI", "Label"]
ef_df.columns = ["Patient", "EF"]

# merge
df = pd.merge(msi_df, ef_df, on="Patient")

# keep only required columns
df = df[["Patient", "MSI", "EF"]]

# save
df.to_csv(out_path, index=False)

print("\nCombined dataset:")
print(df.head())
