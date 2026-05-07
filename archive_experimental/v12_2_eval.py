import pandas as pd

df_pred = pd.read_csv("outputs_v12_2/results.csv")
df_true = pd.read_csv("outputs/ef_results.csv")

df = pd.merge(df_pred, df_true, on="Patient")

df["Error"] = abs(df["EF_est"] - df["EF"])

print("\n=== V12.2 PERFORMANCE ===")
print("Mean Error:", round(df["Error"].mean(),4))
