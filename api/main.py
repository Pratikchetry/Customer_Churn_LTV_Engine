"""
main.py
────────────────────────────────────────────────────
Customer Churn & LTV Prediction API

Endpoints:
    GET  /health
    GET  /predict/{customer_id}
    POST /predict
    GET  /segment/summary

Usage:
    cd api && uvicorn main:app --reload
────────────────────────────────────────────────────
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
import sys
import os
from datetime import datetime

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../src"
    )
)
from feature_engineering import create_features
from predict import predict_customer, get_ltv_thresholds

# ── App Instance ──────────────────────────────────────
app = FastAPI(
    title       = "Customer Churn & LTV API",
    description = "Predict churn risk, lifetime value "
                  "and customer segment",
    version     = "1.0.0",
)

# ── Paths ─────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CHURN_MODEL  = os.path.join(BASE_DIR, "../models/churn_model.pkl")
LTV_MODEL    = os.path.join(BASE_DIR, "../models/ltv_model.pkl")
DATA_PATH    = os.path.join(BASE_DIR, "../data/ecommerce_user_segmentation.csv")
SEGMENTS_PATH = os.path.join(BASE_DIR, "../data/customer_segments.csv")

# ── Load Models Once At Startup ───────────────────────
churn_bundle  = joblib.load(CHURN_MODEL)
ltv_bundle    = joblib.load(LTV_MODEL)
churn_model   = churn_bundle["model"]
ltv_model     = ltv_bundle["model"]
churn_feats   = churn_bundle["features"]
ltv_feats     = ltv_bundle["features"]

print("✅ Models loaded at startup")


# ── Request Schema ────────────────────────────────────
class CustomerFeatures(BaseModel):
    Customer_ID          : Optional[str] = "UNKNOWN"
    Recency              : float
    Frequency            : float
    Monetary             : float
    Avg_Order_Value      : float
    Session_Count        : float
    Avg_Session_Duration : float
    Pages_Viewed         : float
    Clicks               : float
    Campaign_Response    : float
    Wishlist_Adds        : float
    Cart_Abandon_Rate    : float
    Returns              : float

    class Config:
        json_schema_extra = {
            "example": {
                "Customer_ID"         : "CUST00001",
                "Recency"             : 10,
                "Frequency"           : 45,
                "Monetary"            : 5400.0,
                "Avg_Order_Value"     : 120.0,
                "Session_Count"       : 89,
                "Avg_Session_Duration": 25.3,
                "Pages_Viewed"        : 12,
                "Clicks"              : 34,
                "Campaign_Response"   : 1,
                "Wishlist_Adds"       : 8,
                "Cart_Abandon_Rate"   : 0.15,
                "Returns"             : 2,
            }
        }


# ── Response Schema ───────────────────────────────────
class PredictionResponse(BaseModel):
    customer_id      : str
    churn_probability: float
    churn_risk       : str
    predicted_ltv    : float
    ltv_tier         : str
    ltv_low          : float
    ltv_high         : float
    risk_score       : float
    segment          : str
    segment_icon     : str
    actions          : List[str]
    timestamp        : str


class HealthResponse(BaseModel):
    status       : str
    version      : str
    models_loaded: bool
    timestamp    : str


class SegmentSummary(BaseModel):
    segment          : str
    count            : int
    percentage       : float
    avg_ltv          : float
    total_ltv        : float
    avg_risk         : float


class SegmentResponse(BaseModel):
    total_customers   : int
    total_revenue     : float
    revenue_at_risk   : float
    pct_revenue_at_risk: float
    segments          : List[SegmentSummary]
    timestamp         : str


# ── Endpoints ─────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"]
)
def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status        = "healthy",
        version       = "1.0.0",
        models_loaded = True,
        timestamp     = datetime.now().isoformat(),
    )


@app.get(
    "/predict/{customer_id}",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
def predict_by_id(customer_id: str):
    df_raw = pd.read_csv(DATA_PATH)
    match  = df_raw[
        df_raw["Customer_ID"] == customer_id
    ]

    if len(match) == 0:
        raise HTTPException(
            status_code = 404,
            detail      = f"Customer {customer_id} "
                          f"not found"
        )

    features = match.iloc[0].to_dict()
    result   = predict_customer(features)

    return PredictionResponse(
        customer_id       = result["customer_id"],
        churn_probability = result["churn_prob"],
        churn_risk        = result["churn_label"],
        predicted_ltv     = result["predicted_ltv"],
        ltv_tier          = result["ltv_tier"],
        ltv_low           = result["ltv_low"],
        ltv_high          = result["ltv_high"],
        risk_score        = result["risk_score"],
        segment           = result["segment"],
        segment_icon      = result["segment_icon"],
        actions           = result["actions"],
        timestamp         = datetime.now().isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
def predict_by_features(customer: CustomerFeatures):
    features = customer.model_dump()
    result   = predict_customer(features)

    return PredictionResponse(
        customer_id       = result["customer_id"],
        churn_probability = result["churn_prob"],
        churn_risk        = result["churn_label"],
        predicted_ltv     = result["predicted_ltv"],
        ltv_tier          = result["ltv_tier"],
        ltv_low           = result["ltv_low"],
        ltv_high          = result["ltv_high"],
        risk_score        = result["risk_score"],
        segment           = result["segment"],
        segment_icon      = result["segment_icon"],
        actions           = result["actions"],
        timestamp         = datetime.now().isoformat(),
    )



@app.get(
    "/segment/summary",
    response_model=SegmentResponse,
    tags=["Segmentation"]
)
def segment_summary():
    """
    Return segment distribution summary
    from the latest segmentation run.
    """
    if not os.path.exists(SEGMENTS_PATH):
        raise HTTPException(
            status_code = 404,
            detail      = "Segments file not found. "
                          "Run customer_segmentation.py first."
        )

    df = pd.read_csv(SEGMENTS_PATH)

    segment_order = [
        "Champion",
        "At-Risk VIP",
        "Promising",
        "Vulnerable",
        "Hibernating",
        "Losing Customer",
    ]

    summary = df.groupby("Segment").agg(
        count     = ("Customer_ID",    "count"),
        avg_ltv   = ("Predicted_LTV",  "mean"),
        total_ltv = ("Predicted_LTV",  "sum"),
        avg_risk  = ("Composite_Risk_Score", "mean"),
    ).round(2)

    segments = []
    for seg in segment_order:
        if seg in summary.index:
            row = summary.loc[seg]
            segments.append(SegmentSummary(
                segment    = seg,
                count      = int(row["count"]),
                percentage = round(
                    row["count"] / len(df) * 100, 1
                ),
                avg_ltv    = float(row["avg_ltv"]),
                total_ltv  = float(row["total_ltv"]),
                avg_risk   = float(row["avg_risk"]),
            ))

    total_revenue   = df["Predicted_LTV"].sum()
    at_risk_segs    = [
        "At-Risk VIP", "Vulnerable", "Losing Customer"
    ]
    revenue_at_risk = df[
        df["Segment"].isin(at_risk_segs)
    ]["Predicted_LTV"].sum()

    return SegmentResponse(
        total_customers    = len(df),
        total_revenue      = round(total_revenue, 2),
        revenue_at_risk    = round(revenue_at_risk, 2),
        pct_revenue_at_risk= round(
            revenue_at_risk / total_revenue * 100, 1
        ),
        segments           = segments,
        timestamp          = datetime.now().isoformat(),
    )
