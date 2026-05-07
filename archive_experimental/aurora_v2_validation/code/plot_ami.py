import pandas as pd
import matplotlib.pyplot as plt
import os

# load
df = pd.read_csv("aurora_v2_validation/outputs/ami_results.csv")

# output folder
out_dir = "aurora_v2_validation/outputs/plots"
os.makedirs(out_dir, exist_ok=True)

# plot
plt.figure()
plt.scatter(df["AMI"], df["EF"])
plt.xlabel("AMI")
plt.ylabel("EF")
plt.title("AMI vs EF")
plt.grid()

# SAVE
save_path = os.path.join(out_dir, "ami_vs_ef.png")
plt.savefig(save_path)

# SHOW (optional)
plt.show()

print(f"\nSaved AMI plot at: {save_path}")
