import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


df = pd.read_csv("aurora_v2_validation/outputs/v5_combined.csv")

# binary label
df["label"] = df["EF"].apply(lambda x: 1 if x > 0.35 else 0)

print("\nClass distribution:")
print(df["label"].value_counts())

X = df[[
    "ENERGY",
    "ACTIVE_AREA",
    "SPATIAL_STD",
    "R_TL", "R_TR", "R_BL", "R_BR"
]]

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\nWeights:")
print(model.coef_)
