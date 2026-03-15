"""
predict.py
────────────────────────────────────────────────────
Single Customer Prediction Pipeline

Accepts a single customer record and returns:
  - Churn probability + risk level
  - Predicted lifetime value + LTV tier
  - Composite risk score
  - Customer segment
  - Recommended retention action

Modes:
  1. Function call → predict_customer(features_dict)
  2. Command line  → python predict.py --customer_id X
  3. Random sample → python predict.py --random

Usage:
    cd src && python predict.py --random
    cd src && python predict.py --customer_id CUST_0001
────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
import os
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

sys.path.append(str(CURRENT_DIR))
from feature_engineering import create_features

# ── Constants ─────────────────────────────────────────
CHURN_MODEL = PROJECT_ROOT / "models" / "churn_model.pkl"
LTV_MODEL = PROJECT_ROOT / "models" / "ltv_model.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "ecommerce_user_segmentation.csv"

LTV_LOW_PCT = 33
LTV_HIGH_PCT = 67
RISK_THRESHOLD = 0.75

# ── Precomputed LTV thresholds from full dataset ──────
# These are fixed from training data percentiles
# and must remain consistent across all predictions
def get_ltv_thresholds():
    df_raw   = pd.read_csv(DATA_PATH)
    df       = create_features(df_raw.copy())
    ltv_bundle = joblib.load(LTV_MODEL)
    ltv_model  = ltv_bundle["model"]
    ltv_feats  = ltv_bundle["features"]
    ltv_preds  = ltv_model.predict(df[ltv_feats])
    return (
        np.percentile(ltv_preds, LTV_LOW_PCT),
        np.percentile(ltv_preds, LTV_HIGH_PCT),
    )


# ── Risk Score Normalisation Bounds ───────────────────
# Fixed from full dataset to ensure consistency
def get_norm_bounds():
    df_raw  = pd.read_csv(DATA_PATH)
    df      = create_features(df_raw.copy())
    return {
        "churn_min"  : 0.0,
        "churn_max"  : 1.0,
        "recency_min": df["Recency"].min(),
        "recency_max": df["Recency"].max(),
        "abandon_min": df["Cart_Abandon_Rate"].min(),
        "abandon_max": df["Cart_Abandon_Rate"].max(),
    }


# ── Segment Metadata ──────────────────────────────────
SEGMENT_ICONS = {
    "Champion"       : "🟢",
    "At-Risk VIP"    : "🟡",
    "Promising"      : "🔵",
    "Vulnerable"     : "🟠",
    "Hibernating"    : "⚪",
    "Losing Customer": "🔴",
}

SEGMENT_ACTIONS = {
    "Champion": [
        "VIP loyalty programme — reward tenure",
        "Exclusive early access to new products",
        "Proactive relationship management",
        "Do NOT discount — protects margin",
    ],
    "At-Risk VIP": [
        "Immediate personalised outreach",
        "Premium retention offer — $50 budget",
        "Address cart abandonment friction",
        "Re-engage wishlist with price alerts",
    ],
    "Promising": [
        "Upsell to higher order value products",
        "Frequency incentives — buy 3 get 1",
        "Wishlist-based recommendations",
        "Convert to Champion tier focus",
    ],
    "Vulnerable": [
        "Standard retention offer — $25 budget",
        "Session re-engagement campaign",
        "Personalised product recommendations",
        "Monitor risk score weekly",
    ],
    "Hibernating": [
        "Automated low-touch email nurture",
        "Seasonal reactivation campaigns",
        "No significant budget allocation",
        "Monitor for segment migration",
    ],
    "Losing Customer": [
        "Low cost email win-back — $5 budget",
        "Single discount offer — last attempt",
        "Accept churn if no response",
        "Analyse exit patterns for improvement",
    ],
}


# ── Core Prediction Function ──────────────────────────
def predict_customer(features: dict) -> dict:
    """
    Predict churn risk, LTV and segment for a
    single customer.

    Args:
        features: dict of raw feature values
                  (pre-feature-engineering)

    Returns:
        dict with full prediction results
    """
    # Load models
    churn_bundle = joblib.load(CHURN_MODEL)
    ltv_bundle   = joblib.load(LTV_MODEL)
    churn_model  = churn_bundle["model"]
    ltv_model    = ltv_bundle["model"]
    churn_feats  = churn_bundle["features"]
    ltv_feats    = ltv_bundle["features"]

    # Build single-row dataframe + engineer features
    df_input = pd.DataFrame([features])
    df_eng   = create_features(df_input.copy())

    # Predictions
    churn_prob = float(
        churn_model.predict_proba(
            df_eng[churn_feats]
        )[0, 1]
    )
    ltv_pred = float(
        ltv_model.predict(df_eng[ltv_feats])[0]
    )

    # LTV tier
    ltv_low, ltv_high = get_ltv_thresholds()
    if ltv_pred >= ltv_high:
        ltv_tier = "High LTV"
    elif ltv_pred >= ltv_low:
        ltv_tier = "Mid LTV"
    else:
        ltv_tier = "Low LTV"

    # Composite risk score
    bounds = get_norm_bounds()

    def norm(val, mn, mx):
        return (val - mn) / (mx - mn) \
               if mx > mn else 0.0

    churn_norm  = norm(
        churn_prob,
        bounds["churn_min"],
        bounds["churn_max"]
    )
    recency_norm = norm(
        features.get("Recency", 0),
        bounds["recency_min"],
        bounds["recency_max"]
    )
    abandon_norm = norm(
        features.get("Cart_Abandon_Rate", 0),
        bounds["abandon_min"],
        bounds["abandon_max"]
    )

    risk_score = round(
        0.40 * churn_norm +
        0.30 * recency_norm +
        0.30 * abandon_norm,
        4
    )

    # Churn risk label
    if churn_prob >= 0.70:
        churn_label = "🔴 VERY HIGH"
    elif churn_prob >= 0.30:
        churn_label = "🟠 HIGH"
    elif churn_prob >= 0.10:
        churn_label = "🟡 MODERATE"
    else:
        churn_label = "🟢 LOW"

    # Segment — use risk score vs tier
    # For single prediction use absolute
    # risk score thresholds since we cannot
    # compute within-tier percentile for
    # a single customer
    high_risk = risk_score >= 0.20

    if ltv_tier == "High LTV":
        segment = "At-Risk VIP" if high_risk \
                  else "Champion"
    elif ltv_tier == "Mid LTV":
        segment = "Vulnerable"  if high_risk \
                  else "Promising"
    else:
        segment = "Losing Customer" if high_risk \
                  else "Hibernating"

    return {
        "customer_id"  : features.get(
            "Customer_ID", "UNKNOWN"
        ),
        "churn_prob"   : round(churn_prob, 4),
        "churn_label"  : churn_label,
        "predicted_ltv": round(ltv_pred, 2),
        "ltv_tier"     : ltv_tier,
        "ltv_low"      : round(ltv_low, 2),
        "ltv_high"     : round(ltv_high, 2),
        "risk_score"   : risk_score,
        "segment"      : segment,
        "segment_icon" : SEGMENT_ICONS[segment],
        "actions"      : SEGMENT_ACTIONS[segment],
    }


# ── Print Prediction Report ───────────────────────────
def print_report(result: dict):
    """Print formatted prediction report."""
    print("\n" + "=" * 55)
    print("   CUSTOMER PREDICTION REPORT")
    print("=" * 55)
    print(f"  Customer ID        : "
          f"{result['customer_id']}")
    print("  " + "-" * 51)
    print(f"  Churn Probability  : "
          f"{result['churn_prob']:.4f}")
    print(f"  Churn Risk         : "
          f"{result['churn_label']}")
    print("  " + "-" * 51)
    print(f"  Predicted LTV      : "
          f"${result['predicted_ltv']:,.2f}")
    print(f"  LTV Tier           : "
          f"{result['ltv_tier']}")
    print(f"  LTV Thresholds     : "
          f"Low < ${result['ltv_low']:,.2f} | "
          f"High > ${result['ltv_high']:,.2f}")
    print("  " + "-" * 51)
    print(f"  Composite Risk     : "
          f"{result['risk_score']:.4f}")
    print(f"  Segment            : "
          f"{result['segment_icon']} "
          f"{result['segment']}")
    print("  " + "-" * 51)
    print("  RECOMMENDED ACTIONS:")
    for action in result["actions"]:
        print(f"    → {action}")
    print("=" * 55)


# ── CLI Entry Point ───────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict churn + LTV for "
                    "a single customer"
    )
    parser.add_argument(
        "--customer_id",
        type=str,
        help="Customer ID from dataset"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Pick a random customer from dataset"
    )
    args = parser.parse_args()

    # Load dataset
    df_raw = pd.read_csv(DATA_PATH)

    if args.random:
        row = df_raw.sample(1).iloc[0]
        print(f"\n  Random customer selected: "
              f"{row['Customer_ID']}")
    elif args.customer_id:
        matches = df_raw[
            df_raw["Customer_ID"] == args.customer_id
        ]
        if len(matches) == 0:
            print(f"  Customer {args.customer_id} "
                  f"not found.")
            sys.exit(1)
        row = matches.iloc[0]
    else:
        # Default — pick first customer
        row = df_raw.iloc[0]

    features = row.to_dict()
    result   = predict_customer(features)
    print_report(result)