import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("aurora_v2_validation/outputs/final_batch/results.csv")

plt.hist(df["MSI"], bins=20)
plt.title("MSI Distribution")
plt.xlabel("MSI")
plt.ylabel("Count")
plt.show()
