import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs_v17/results.csv")

print("\n=== BASIC STATS ===")
print(df.describe())

print("\n=== STATE COUNTS ===")
print(df["State"].value_counts())

print("\n=== RANGE ===")
print("Min:", df["Deviation"].min())
print("Max:", df["Deviation"].max())
print("Std:", df["Deviation"].std())

# -------------------------
# HISTOGRAM
# -------------------------
plt.figure()
plt.hist(df["Deviation"], bins=20)
plt.title("Deviation Distribution")
plt.xlabel("Deviation")
plt.ylabel("Count")
plt.savefig("outputs_v17/deviation_hist.png")

# -------------------------
# SCATTER (index vs deviation)
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
        range(len(subset)),
        subset["Deviation"],
        color=colors[state],
        label=state
    )

plt.title("Deviation by Patient")
plt.legend()
plt.savefig("outputs_v17/deviation_scatter.png")

print("\nPlots saved in outputs_v17/")
