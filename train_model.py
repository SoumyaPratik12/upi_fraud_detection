import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os

# Load dataset
print("Loading PaySim dataset...")
df = pd.read_csv("paysim_data.csv")

# Feature engineering
print("Engineering features...")
df["hour"] = df["step"] % 24
df["amount_log"] = np.log1p(df["amount"])
df["balance_diff_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

# Additional fraud indicators
df["balance_change_pct_orig"] = df["balance_diff_orig"] / (df["oldbalanceOrg"] + 1)
df["balance_change_pct_dest"] = df["balance_diff_dest"] / (df["oldbalanceDest"] + 1)
df["amount_hour_interaction"] = df["amount_log"] * df["hour"]
df["balance_ratio_interaction"] = df["balance_diff_orig"] * df["amount_to_balance_ratio"]
df["is_merchant_dest"] = df["nameDest"].str.startswith("M").astype(int)
df["is_customer_dest"] = df["nameDest"].str.startswith("C").astype(int)
df["high_amount_flag"] = (df["amount"] > df["amount"].quantile(0.95)).astype(int)
df["unusual_hour_flag"] = ((df["hour"] >= 23) | (df["hour"] <= 4)).astype(int)
df["balance_drained_flag"] = (df["newbalanceOrig"] == 0).astype(int)

# Encode transaction type
type_mapping = {"CASH_OUT": 0, "PAYMENT": 1, "CASH_IN": 2, "TRANSFER": 3, "DEBIT": 4}
df["type_encoded"] = df["type"].map(type_mapping)

# Select features
features = [
    "type_encoded",
    "amount_log",
    "hour",
    "balance_diff_orig",
    "balance_diff_dest",
    "amount_to_balance_ratio",
    "oldbalanceOrg",
    "newbalanceOrig",
    "balance_change_pct_orig",
    "balance_change_pct_dest",
    "amount_hour_interaction",
    "balance_ratio_interaction",
    "is_merchant_dest",
    "is_customer_dest",
    "high_amount_flag",
    "unusual_hour_flag",
    "balance_drained_flag",
]

X = df[features]
y = df["isFraud"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate class imbalance ratio
fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class imbalance ratio: {fraud_ratio:.0f}:1")

# Train XGBoost model
print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=fraud_ratio,
    eval_metric="auc",
    random_state=42,
)

model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

y_pred = (y_pred_proba > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

# Save model and feature list
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fraud_model.joblib")
joblib.dump(features, "model/feature_names.joblib")
print("\nModel saved to model/fraud_model.joblib")