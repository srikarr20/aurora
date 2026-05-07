import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("outputs_final/results.csv")

print("\n=== BASIC STATS ===")
print(df.describe())

# -------------------------
# Histogram: MeanDeviation
# -------------------------
plt.figure()
plt.hist(df["MeanDeviation"], bins=20)
plt.title("Mean Deviation Distribution")
plt.xlabel("MeanDeviation")
plt.ylabel("Count")
plt.savefig("outputs_final/deviation_hist.png")

# -------------------------
# Histogram: Trend
# -------------------------
plt.figure()
plt.hist(df["Trend"], bins=20)
plt.title("Trend Distribution")
plt.xlabel("Trend")
plt.ylabel("Count")
plt.savefig("outputs_final/trend_hist.png")

# -------------------------
# Scatter plot
# -------------------------
plt.figure()

colors = {
    "STABLE": "green",
    "UNSTABLE": "orange",
    "DYSFUNCTIONAL": "red"
}

for state in df["State"].unique():
    subset = df[df["State"] == state]
    plt.scatter(
        subset["MeanDeviation"],
        subset["Trend"],
        label=state,
        color=colors.get(state, "blue")
    )

plt.xlabel("MeanDeviation")
plt.ylabel("Trend")
plt.title("Deviation vs Trend")
plt.legend()
plt.savefig("outputs_final/dev_vs_trend.png")

# -------------------------
# Group stats
# -------------------------
print("\n=== GROUP STATS ===")
print(df.groupby("State")[["MeanDeviation", "Trend"]].mean())

# -------------------------
# Correlation
# -------------------------
print("\n=== CORRELATION ===")
print(df[["MeanDeviation", "Trend"]].corr())

print("\nPlots saved in outputs_final/")
