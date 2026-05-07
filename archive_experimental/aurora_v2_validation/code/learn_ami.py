import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("aurora_v2_validation/outputs/v5_combined.csv")

# features
X = df[["ENERGY", "ACTIVE_AREA", "SPATIAL_STD"]]
y = df["EF"]

# train regression
model = LinearRegression()
model.fit(X, y)

# predicted AMI
df["AMI_learned"] = model.predict(X)

print("\nWeights:", model.coef_)

# plot
plt.scatter(df["AMI_learned"], df["EF"])
plt.xlabel("Learned AMI")
plt.ylabel("EF")
plt.title("Learned AMI vs EF")
plt.grid()
plt.show()
