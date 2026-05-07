import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("aurora_v2_validation/outputs/combined.csv")

print("\nLoaded Data:")
print(df.head())


# =========================
# BINARY LABELS
# =========================
def label_from_ef(ef):
    if ef > 0.35:
        return 1   # NORMAL
    else:
        return 0   # ABNORMAL

df["label"] = df["EF"].apply(label_from_ef)


# =========================
# CHECK CLASS BALANCE
# =========================
print("\nClass distribution:")
print(df["label"].value_counts())


# =========================
# FEATURES
# =========================
X = df[["MSI"]]
y = df["label"]


# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# =========================
# TRAIN MODEL
# =========================
model = LogisticRegression()
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
# MODEL INTERPRETATION
# =========================
print("\nModel Coefficients:")
print("Weight:", model.coef_)
print("Bias:", model.intercept_)
