# UPI Fraud Detection API - Setup Guide

## Overview
This project is a fraud risk scoring API for UPI transactions built with FastAPI and XGBoost.

## Installation

### For Production (API Runtime Only)
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements-prod.txt
```

### For Development (Model Training + API)
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements-dev.txt
python train_model.py  # Train the model
python -m uvicorn main:app --reload
```

## File Structure
- `main.py` - FastAPI application for fraud scoring
- `train_model.py` - XGBoost model training script
- `test_model.py` - Direct model testing (debug)
- `model/` - Saved XGBoost model and feature names
- `requirements.txt` - All dependencies (dev + prod)
- `requirements-prod.txt` - Minimal runtime dependencies
- `requirements-dev.txt` - Development + training dependencies

## API Endpoint
- `POST /v1/score` - Score a transaction for fraud risk
- `GET /health` - Health check

## Minimizing venv Size

### Option 1: Use Production Requirements Only
For API-only deployment, use `requirements-prod.txt`:
```bash
pip install -r requirements-prod.txt
```
**Size reduction**: ~50% smaller (excludes pandas, scikit-learn)

### Option 2: Docker Production Image
Use a slim base image and install only production requirements:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 3: Clean Cache
After installing, clean pip cache to save disk space:
```bash
pip cache purge
```

## Features Added
- 17 engineered fraud indicators
- Balance change percentages
- Amount-hour interaction terms
- Transaction type flags
- Balance drain detection
- XGBoost model with scale_pos_weight for class imbalance

## Model Performance
- ROC-AUC: ~0.96
- Detects unusual patterns: high amounts, odd hours, balance drainage
