import pandas as pd

df_pred = pd.read_csv("outputs_v14/results.csv")
df_true = pd.read_csv("outputs/ef_results.csv")

df = pd.merge(df_pred, df_true, on="Patient")

df["Error"] = abs(df["EF_est"] - df["EF"])

print("\n=== V14 PERFORMANCE ===")
print("Mean Error:", round(df["Error"].mean(),4))
