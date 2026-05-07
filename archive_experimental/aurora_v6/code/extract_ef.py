import pandas as pd

# load your existing file
df = pd.read_csv("../aurora_v2_validation/outputs/combined.csv")

# keep only needed columns
df_out = df[["Patient", "EF"]]

# remove duplicates (important)
df_out = df_out.drop_duplicates(subset=["Patient"])

# save clean EF file
df_out.to_csv("outputs/ef_results.csv", index=False)

print("Saved: outputs/ef_results.csv")
print(df_out.head())
