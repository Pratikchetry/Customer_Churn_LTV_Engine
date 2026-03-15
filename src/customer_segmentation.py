"""
customer_segmentation.py
────────────────────────────────────────────────────
Customer Value Segmentation Pipeline

Combineing churn probability and predicted LTV
to assign all customers to one of six behavioural
segments using a composite risk score and
within-tier percentile ranking.

Segments:
    🟢 Champion        → High LTV, Low Risk
    🟡 At-Risk VIP     → High LTV, High Risk
    🔵 Promising       → Mid LTV,  Low Risk
    🟠 Vulnerable      → Mid LTV,  High Risk
    ⚪ Hibernating     → Low LTV,  Low Risk
    🔴 Losing Customer → Low LTV,  High Risk

Output:
    data/customer_segments.csv

Usage:
    cd src && python customer_segmentation.py
────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)
from feature_engineering import create_features

# ── Constants ─────────────────────────────────────────
DATA_PATH    = "../data/ecommerce_user_segmentation.csv"
CHURN_MODEL  = "../models/churn_model.pkl"
LTV_MODEL    = "../models/ltv_model.pkl"
OUTPUT_PATH  = "../data/customer_segments.csv"

LTV_LOW_PCT  = 33
LTV_HIGH_PCT = 67
RISK_THRESHOLD = 0.75    # top 25% within tier

SEGMENT_ORDER = [
    "Champion",
    "At-Risk VIP",
    "Promising",
    "Vulnerable",
    "Hibernating",
    "Losing Customer",
]

RECOVERY_PARAMS = {
    "At-Risk VIP"    : {"cost": 50,  "rate": 0.45},
    "Vulnerable"     : {"cost": 25,  "rate": 0.35},
    "Losing Customer": {"cost":  5,  "rate": 0.15},
}


# ── Step 1 — Load Models ──────────────────────────────
def load_models():
    """Loading churn and LTV production models."""
    churn_bundle = joblib.load(CHURN_MODEL)
    ltv_bundle   = joblib.load(LTV_MODEL)
    return (
        churn_bundle["model"],
        ltv_bundle["model"],
        churn_bundle["features"],
        ltv_bundle["features"],
    )


# ── Step 2 — Load Data ────────────────────────────────
def load_data():
    """Loading and engineering features from raw dataset."""
    df_raw = pd.read_csv(DATA_PATH)
    df     = create_features(df_raw.copy())
    return df


# ── Step 3 — Generate Predictions ────────────────────
def generate_predictions(df, churn_model, ltv_model,
                         churn_feats, ltv_feats):
    """Generating churn probability and LTV for all
    customers."""
    churn_probs = churn_model.predict_proba(
        df[churn_feats]
    )[:, 1]
    ltv_preds   = ltv_model.predict(df[ltv_feats])

    pred_df = pd.DataFrame({
        "Customer_ID"   : df["Customer_ID"],
        "Churn_Prob"    : churn_probs.round(4),
        "Predicted_LTV" : ltv_preds.round(2),
        "Recency"       : df["Recency"].values,
        "Cart_Abandon_Rate":
            df["Cart_Abandon_Rate"].values,
    })
    return pred_df


# ── Step 4 — Compute Composite Risk Score ────────────
def compute_risk_score(pred_df):
    """Computing normalised composite risk score from
    churn probability, recency and cart abandonment."""

    def norm(series):
        return (series - series.min()) / (
            series.max() - series.min()
        )

    pred_df["Churn_norm"]  = norm(pred_df["Churn_Prob"])
    pred_df["Recency_norm"] = norm(pred_df["Recency"])
    pred_df["Abandon_norm"] = norm(
        pred_df["Cart_Abandon_Rate"]
    )

    pred_df["Risk_Score"] = (
        0.40 * pred_df["Churn_norm"]  +
        0.30 * pred_df["Recency_norm"] +
        0.30 * pred_df["Abandon_norm"]
    ).round(4)

    return pred_df


# ── Step 5 — Assign Segments ──────────────────────────
def assign_segments(pred_df):
    """Assign customers to segments using within-tier
    percentile ranking of composite risk score."""

    ltv_low  = np.percentile(
        pred_df["Predicted_LTV"], LTV_LOW_PCT
    )
    ltv_high = np.percentile(
        pred_df["Predicted_LTV"], LTV_HIGH_PCT
    )

    pred_df["LTV_Tier"] = pd.cut(
        pred_df["Predicted_LTV"],
        bins=[-np.inf, ltv_low, ltv_high, np.inf],
        labels=["Low LTV", "Mid LTV", "High LTV"]
    )

    pred_df["Risk_Percentile"] = pred_df.groupby(
        "LTV_Tier"
    )["Risk_Score"].rank(pct=True).round(4)

    def assign(row):
        high_risk = row["Risk_Percentile"] >= RISK_THRESHOLD
        tier      = row["LTV_Tier"]
        if tier == "High LTV":
            return "At-Risk VIP"     if high_risk \
                   else "Champion"
        elif tier == "Mid LTV":
            return "Vulnerable"      if high_risk \
                   else "Promising"
        else:
            return "Losing Customer" if high_risk \
                   else "Hibernating"

    pred_df["Segment"] = pred_df.apply(assign, axis=1)
    return pred_df, ltv_low, ltv_high


# ── Step 6 — Generate Report ──────────────────────────
def generate_report(pred_df):
    """Printing segment summary report."""
    summary = pred_df.groupby("Segment").agg(
        Count          = ("Customer_ID",    "count"),
        Avg_Risk       = ("Risk_Score",     "mean"),
        Avg_Churn_Prob = ("Churn_Prob",     "mean"),
        Avg_LTV        = ("Predicted_LTV",  "mean"),
        Total_LTV      = ("Predicted_LTV",  "sum"),
    ).round(2).reindex(SEGMENT_ORDER).fillna(0)

    summary["Pct"] = (
        summary["Count"] / len(pred_df) * 100
    ).round(1)

    icons = {
        "Champion"       : "🟢",
        "At-Risk VIP"    : "🟡",
        "Promising"      : "🔵",
        "Vulnerable"     : "🟠",
        "Hibernating"    : "⚪",
        "Losing Customer": "🔴",
    }

    print("=" * 75)
    print("   CUSTOMER SEGMENTATION REPORT")
    print("=" * 75)
    print(
        f"\n  {'Segment':<22} {'Count':>7} {'%':>6}"
        f" {'Avg Risk':>10} {'Avg Churn':>10}"
        f" {'Avg LTV':>12} {'Total LTV':>14}"
    )
    print("  " + "-" * 83)

    for seg in SEGMENT_ORDER:
        row  = summary.loc[seg]
        icon = icons[seg]
        print(
            f"  {icon} {seg:<20}"
            f" {int(row['Count']):>7,}"
            f" {row['Pct']:>5.1f}%"
            f" {row['Avg_Risk']:>10.4f}"
            f" {row['Avg_Churn_Prob']:>10.4f}"
            f" ${row['Avg_LTV']:>10,.2f}"
            f" ${row['Total_LTV']:>12,.2f}"
        )

    total_ltv   = pred_df["Predicted_LTV"].sum()
    at_risk_ltv = pred_df[
        pred_df["Segment"].isin(
            ["At-Risk VIP", "Vulnerable",
             "Losing Customer"]
        )
    ]["Predicted_LTV"].sum()

    print("\n" + "=" * 75)
    print(f"  Total Revenue Modelled  : "
          f"${total_ltv:>14,.2f}")
    print(f"  Revenue At Risk         : "
          f"${at_risk_ltv:>14,.2f}")
    print(f"  % Revenue At Risk       : "
          f"{at_risk_ltv/total_ltv*100:>12.1f}%")
    print("=" * 75)

    return summary


# ── Step 7 — Recovery Simulation ─────────────────────
def run_recovery_simulation(summary):
    """Simulating revenue recovery ROI per at-risk
    segment."""
    print("\n" + "=" * 90)
    print("   REVENUE RECOVERY SIMULATION")
    print("=" * 90)
    print(
        f"\n  {'Segment':<20} {'Customers':>10}"
        f" {'At Risk $':>14} {'Campaign $':>12}"
        f" {'Recovered $':>14} {'Net ROI $':>14}"
        f" {'ROI %':>8}"
    )
    print("  " + "-" * 88)

    total_cost = total_rec = total_roi = 0

    for seg, params in RECOVERY_PARAMS.items():
        count         = int(summary.loc[seg, "Count"])
        total_ltv     = summary.loc[seg, "Total_LTV"]
        campaign_cost = count * params["cost"]
        recovered     = total_ltv * params["rate"]
        net_roi       = recovered - campaign_cost
        roi_pct       = net_roi / campaign_cost * 100

        total_cost += campaign_cost
        total_rec  += recovered
        total_roi  += net_roi

        print(
            f"  {seg:<20}"
            f" {count:>10,}"
            f" ${total_ltv:>13,.2f}"
            f" ${campaign_cost:>11,.2f}"
            f" ${recovered:>13,.2f}"
            f" ${net_roi:>13,.2f}"
            f" {roi_pct:>7.0f}%"
        )

    print("  " + "-" * 88)
    print(
        f"  {'TOTAL':<20}"
        f" {sum(int(summary.loc[s, 'Count']) for s in RECOVERY_PARAMS):>10,}"
        f" {'':>14}"
        f" ${total_cost:>11,.2f}"
        f" ${total_rec:>13,.2f}"
        f" ${total_roi:>13,.2f}"
        f" {total_roi/total_cost*100:>7.0f}%"
    )
    print("=" * 90)


# ── Step 8 — Export CSV ───────────────────────────────
def export_segments(pred_df):
    """Exporting segmented customer list to CSV."""
    export_df = pred_df[[
        "Customer_ID",
        "Churn_Prob",
        "Predicted_LTV",
        "Risk_Score",
        "Segment",
    ]].copy()

    export_df.columns = [
        "Customer_ID",
        "Churn_Probability",
        "Predicted_LTV",
        "Composite_Risk_Score",
        "Segment",
    ]

    export_df = export_df.sort_values(
        "Composite_Risk_Score",
        ascending=False
    ).reset_index(drop=True)

    export_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Exported → {OUTPUT_PATH}")
    print(f"  Rows     → {len(export_df):,}")
    print("  customer_segmentation.py COMPLETE ✅")


# ── Main Pipeline ─────────────────────────────────────
if __name__ == "__main__":
    print("=" * 75)
    print("   CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 75)

    print("\n  Step 1 → Loading models...")
    churn_model, ltv_model, \
    churn_feats, ltv_feats = load_models()
    print("  ✅ Models loaded")

    print("  Step 2 → Loading data...")
    df = load_data()
    print(f"  ✅ Data loaded — {len(df):,} customers")

    print("  Step 3 → Generating predictions...")
    pred_df = generate_predictions(
        df, churn_model, ltv_model,
        churn_feats, ltv_feats
    )
    print("  ✅ Predictions generated")

    print("  Step 4 → Computing risk scores...")
    pred_df = compute_risk_score(pred_df)
    print("  ✅ Risk scores computed")

    print("  Step 5 → Assigning segments...")
    pred_df, ltv_low, ltv_high = assign_segments(
        pred_df
    )
    print("  ✅ Segments assigned")

    print("\n  Step 6 → Generating report...")
    summary = generate_report(pred_df)

    print("\n  Step 7 → Running recovery simulation...")
    run_recovery_simulation(summary)

    print("\n  Step 8 → Exporting segments...")
    export_segments(pred_df)
