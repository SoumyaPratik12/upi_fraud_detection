from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
import joblib
import time

# Initialize the FastAPI application with metadata
app = FastAPI(
    title="UPI Fraud Risk Scoring API",
    description="Real-time fraud risk scoring for UPI transactions. Returns risk score, decision, and reason codes.",
    version="1.0.0",
)

# Load model at startup
model = joblib.load("model/fraud_model.joblib")
feature_names = joblib.load("model/feature_names.joblib")

class TransactionRequest(BaseModel):
    amount: float = Field(..., description="Transaction amount in INR", gt=0)
    transaction_type: Literal["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"] = Field(
        ..., description="Type of UPI transaction"
    )
    hour: int = Field(..., description="Hour of transaction (0-23 IST)", ge=0, le=23)
    sender_balance: float = Field(..., description="Sender account balance before transaction", ge=0)
    receiver_balance: float = Field(..., description="Receiver account balance before transaction", ge=0)


class RiskResponse(BaseModel):
    risk_score: int = Field(..., description="Risk score from 0 (safe) to 100 (fraud)")
    decision: Literal["ALLOW", "FLAG", "BLOCK"] = Field(..., description="Recommended action")
    reasons: list[str] = Field(..., description="Reason codes explaining the risk assessment")
    processing_time_ms: float = Field(..., description="Time taken to process in milliseconds")


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/v1/score", response_model=RiskResponse)
def score_transaction(txn: TransactionRequest):
    start_time = time.time()

    # Map transaction type to numeric encoding (same as training)
    type_mapping = {"CASH_OUT": 0, "PAYMENT": 1, "CASH_IN": 2, "TRANSFER": 3, "DEBIT": 4}

    # Engineer the same features the model was trained on
    amount_log = np.log1p(txn.amount)
    newbalance_orig = txn.sender_balance - txn.amount
    balance_diff_orig = newbalance_orig - txn.sender_balance
    newbalance_dest = txn.receiver_balance if txn.transaction_type == "CASH_OUT" else txn.receiver_balance + txn.amount
    balance_diff_dest = newbalance_dest - txn.receiver_balance
    amount_to_balance_ratio = txn.amount / (txn.sender_balance + 1)

    # Additional fraud indicators
    balance_change_pct_orig = balance_diff_orig / (txn.sender_balance + 1)
    balance_change_pct_dest = balance_diff_dest / (txn.receiver_balance + 1)
    amount_hour_interaction = amount_log * txn.hour
    balance_ratio_interaction = balance_diff_orig * amount_to_balance_ratio
    # Assuming CASH_OUT is to customer (common in PaySim), PAYMENT/DEBIT to merchant
    is_merchant_dest = 1 if txn.transaction_type in ["PAYMENT", "DEBIT"] else 0
    is_customer_dest = 1 if txn.transaction_type == "CASH_OUT" else 0
    high_amount_flag = 1 if txn.amount > 50000 else 0  # Using a fixed threshold
    unusual_hour_flag = 1 if (txn.hour >= 23 or txn.hour <= 4) else 0
    balance_drained_flag = 1 if newbalance_orig == 0 else 0

    # Build feature array in the exact order the model expects
    features = np.array([[
        type_mapping[txn.transaction_type],
        amount_log,
        txn.hour,
        balance_diff_orig,
        balance_diff_dest,
        amount_to_balance_ratio,
        txn.sender_balance,
        newbalance_orig,
        balance_change_pct_orig,
        balance_change_pct_dest,
        amount_hour_interaction,
        balance_ratio_interaction,
        is_merchant_dest,
        is_customer_dest,
        high_amount_flag,
        unusual_hour_flag,
        balance_drained_flag,
    ]])

    # Get fraud probability from model and convert to 0-100 score
    # fraud_probability = model.predict_proba(features)[0][1]
    fraud_probability = model.predict_proba(features)[0][1]
    print(f"DEBUG - fraud_probability: {fraud_probability}")
    print(f"DEBUG - features: {features}")
    risk_score = int(fraud_probability * 100)

    # Apply decision thresholds
    if risk_score >= 80:
        decision = "BLOCK"
    elif risk_score >= 50:
        decision = "FLAG"
    else:
        decision = "ALLOW"

    # Generate reason codes based on transaction characteristics
    reasons = []
    if txn.hour >= 23 or txn.hour <= 4:
        reasons.append("unusual_hour")
    if txn.amount > 50000:
        reasons.append("high_amount")
    if amount_to_balance_ratio > 0.8:
        reasons.append("amount_exceeds_balance_threshold")
    if txn.transaction_type in ["CASH_OUT", "TRANSFER"]:
        reasons.append("high_risk_transaction_type")
    if not reasons:
        reasons.append("normal_pattern")

    # Calculate processing time for monitoring
    processing_time = (time.time() - start_time) * 1000

    return RiskResponse(
        risk_score=risk_score,
        decision=decision,
        reasons=reasons,
        processing_time_ms=round(processing_time, 2),
    )