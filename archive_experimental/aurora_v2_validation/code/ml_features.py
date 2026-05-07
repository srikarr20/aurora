import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("aurora_v2_validation/outputs/combined_features.csv")

print("\nLoaded Data:")
print(df.head())


# =========================
# BINARY LABEL
# =========================
df["label"] = df["EF"].apply(lambda x: 1 if x > 0.35 else 0)

print("\nClass distribution:")
print(df["label"].value_counts())


# =========================
# FEATURES
# =========================
X = df[["MSI", "ENERGY", "TEMP_VAR"]]
y = df["label"]


# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =========================
# TRAIN
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# =========================
# PREDICT
# =========================
y_pred = model.predict(X_test)


# =========================
# RESULTS
# =========================
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))


# =========================
# MODEL WEIGHTS
# =========================
print("\nFeature Weights:")
print("MSI, ENERGY, TEMP_VAR →", model.coef_)
