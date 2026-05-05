
import numpy as np
import joblib

# Load the model
model = joblib.load("model/fraud_model.joblib")

# Simulate the high-risk CASH_OUT transaction
features = np.array([[
    0,             # CASH_OUT
    np.log1p(200000),  # amount_log
    2,             # hour (2 AM)
    -200000,       # balance_diff_orig
    0,             # balance_diff_dest for CASH_OUT
    200000 / (210000 + 1),  # amount_to_balance_ratio
    210000,        # sender_balance (oldbalanceOrg)
    10000,         # sender_balance - amount (newbalanceOrig)
    -200000 / (210000 + 1),  # balance_change_pct_orig
    0 / (0 + 1),    # balance_change_pct_dest (receiver_balance=0)
    np.log1p(200000) * 2,  # amount_hour_interaction
    -200000 * (200000 / (210000 + 1)),  # balance_ratio_interaction
    0,             # is_merchant_dest
    1,             # is_customer_dest
    1,             # high_amount_flag (>50000)
    1,             # unusual_hour_flag (2 AM)
    0,             # balance_drained_flag (newbalanceOrig=10000 !=0)
]])

fraud_prob = model.predict_proba(features)[0][1]
risk_score = int(fraud_prob * 100)

print(f"fraud_probability: {fraud_prob}")
print(f"risk_score: {risk_score}")
print(f"feature shape: {features.shape}")
print(f"feature dtypes: {features.dtype}")
