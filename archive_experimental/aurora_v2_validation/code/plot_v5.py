import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("aurora_v2_validation/outputs/v5_combined.csv")

plt.scatter(df["ENERGY"], df["EF"])
plt.xlabel("AURORA Energy")
plt.ylabel("EF")
plt.title("Energy vs EF")
plt.show()

