import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

# ------------------ LOAD DATA ------------------
df = pd.read_csv("loan_approval_dataset.csv")

# Clean column names (remove spaces)
df.columns = df.columns.str.strip()

print("Columns:", df.columns)

# ------------------ DROP ID ------------------
if "loan_id" in df.columns:
    df.drop("loan_id", axis=1, inplace=True)

# ------------------ HANDLE MISSING VALUES ------------------
df.fillna(df.median(numeric_only=True), inplace=True)

# ------------------ ENCODE CATEGORICAL ------------------
le = LabelEncoder()

df['education'] = le.fit_transform(df['education'])
df['self_employed'] = le.fit_transform(df['self_employed'])

# ------------------ TARGET ------------------
# encoding
df['education'] = le.fit_transform(df['education'])
df['self_employed'] = le.fit_transform(df['self_employed'])

# 👉 PUT FIX HERE
# ------------------ TARGET FIX ------------------
df['loan_status'] = df['loan_status'].astype(str).str.strip().str.lower()
df['loan_status'] = df['loan_status'].map({
    "approved": 1,
    "rejected": 0
})
df = df.dropna(subset=['loan_status'])

# then continue
X = df.drop("loan_status", axis=1)
y = df["loan_status"]
# ------------------ FEATURES & TARGET ------------------
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

print("\nFeature Columns:\n", X.columns)

# ------------------ TRAIN TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------ MODEL ------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# ------------------ EVALUATION ------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📊 MODEL PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------ FEATURE IMPORTANCE ------------------
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\n🔥 Feature Importance:\n", importances.sort_values(ascending=False))

# ------------------ SAVE MODEL ------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model trained and saved as model.pkl")