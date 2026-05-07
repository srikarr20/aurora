import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
df_cluster = pd.read_csv("outputs_v17/results_clustered.csv")
df_ef = pd.read_csv("outputs/ef_results.csv")

# =========================
# MERGE ON PATIENT
# =========================
df = pd.merge(df_cluster, df_ef, on="Patient")

print("\n=== SAMPLE DATA ===")
print(df.head())

# =========================
# EF PER CLUSTER
# =========================
print("\n=== EF BY CLUSTER ===")

cluster_means = {}

for c in sorted(df["Cluster"].unique()):
    vals = df[df["Cluster"] == c]["EF"]

    mean_val = vals.mean()
    cluster_means[c] = mean_val

    print(f"\nCluster {c}:")
    print("Mean EF:", round(mean_val, 4))
    print("Std EF:", round(vals.std(), 4))
    print("Min EF:", round(vals.min(), 4))
    print("Max EF:", round(vals.max(), 4))

# =========================
# CLUSTER ORDER (LOW → HIGH EF)
# =========================
print("\n=== CLUSTER ORDER (LOW → HIGH EF) ===")
ordered = sorted(cluster_means, key=cluster_means.get)
print(ordered)

# =========================
# CORRELATION
# =========================
corr = df["Deviation"].corr(df["EF"])

print("\n=== CORRELATION ===")
print("Deviation vs EF:", round(corr, 4))

# =========================
# BOXPLOT
# =========================
plt.figure()

clusters = sorted(df["Cluster"].unique())
data = [df[df["Cluster"] == c]["EF"] for c in clusters]

plt.boxplot(data, labels=[f"C{c}" for c in clusters])
plt.title("EF Distribution per Cluster")
plt.xlabel("Cluster")
plt.ylabel("EF")
plt.savefig("outputs_v17/ef_boxplot.png")

# =========================
# SCATTER: DEVIATION vs EF
# =========================
plt.figure()

colors = ["green", "orange", "red"]

for c in clusters:
    subset = df[df["Cluster"] == c]
    plt.scatter(
        subset["Deviation"],
        subset["EF"],
        color=colors[c],
        label=f"Cluster {c}"
    )

plt.xlabel("Deviation")
plt.ylabel("EF")
plt.title("Deviation vs EF")
plt.legend()
plt.savefig("outputs_v17/deviation_vs_ef.png")

# =========================
# SORTED COMPARISON
# =========================
df_sorted = df.sort_values("Deviation")

plt.figure()
plt.plot(df_sorted["Deviation"].values, label="Deviation")
plt.plot(df_sorted["EF"].values, label="EF")
plt.legend()
plt.title("Sorted Deviation vs EF")
plt.savefig("outputs_v17/sorted_dev_vs_ef.png")

print("\nSaved plots in outputs_v17/")
