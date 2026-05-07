import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("aurora_v2_validation/outputs/v5_combined.csv")

# ACI
df["ACI"] = df["SPATIAL_STD"] - 0.3 * df["ENERGY"]

plt.scatter(df["ACI"], df["EF"])
plt.xlabel("ACI (Coherence)")
plt.ylabel("EF")
plt.title("ACI vs EF")
plt.grid()
plt.show()
