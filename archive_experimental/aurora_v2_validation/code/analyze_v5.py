import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("aurora_v2_validation/outputs/v5_combined.csv")

out_dir = "aurora_v2_validation/outputs/plots"
os.makedirs(out_dir, exist_ok=True)

# =========================
# FUNCTION TO SAVE CLEANLY
# =========================
def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()   # IMPORTANT

# =========================
# PLOTS
# =========================

# 1. ENERGY vs EF
save_plot(df["ENERGY"], df["EF"],
          "AURORA Energy", "EF",
          "Energy vs EF", "energy_vs_ef.png")

# 2. ACTIVE AREA vs EF
save_plot(df["ACTIVE_AREA"], df["EF"],
          "Active Area", "EF",
          "Active Area vs EF", "area_vs_ef.png")

# 3. SPATIAL STD vs EF
save_plot(df["SPATIAL_STD"], df["EF"],
          "Spatial Std", "EF",
          "Spatial Std vs EF", "std_vs_ef.png")

# 4. RATIO
ratio = df["ACTIVE_AREA"] / (df["ENERGY"] + 1e-8)

save_plot(ratio, df["EF"],
          "Area / Energy", "EF",
          "Ratio vs EF", "ratio_vs_ef.png")

print("\nAll plots saved in:", out_dir)
