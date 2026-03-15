import pandas as pd
import numpy as np


def create_features(df):
    """
    Central feature engineering function for
    Customer Churn and LTV prediction pipeline.

    Creates all derived features from raw data.
    Individual modeling scripts select relevant
    features for their specific task.

    Features created:
        Ratio features    → Clicks_per_Page,
                            Wishlist_Conversion
        Churn signals     → Return_Rate,
                            Abandon_Intensity
        LTV target        → LTV
        Churn label       → Churn

    Features intentionally excluded:
        Engagement_Score  → Dropped (0.95 corr
                            with Session_Count —
                            mathematically redundant)
        Revenue_per_Session → Created but dropped
                            in LTV modeling due to
                            indirect data leakage
                            (corr 1.0 with Monetary
                            via Session_Count)

    Args:
        df: Raw DataFrame from
            ecommerce_user_segmentation.csv

    Returns:
        df: DataFrame with engineered features
    """

    # ── Ratio Features ────────────────────────────────
    # Normalises click behavior by pages viewed
    # Removes bias toward customers who view more pages
    df["Clicks_per_Page"] = (
        df["Clicks"] / (df["Pages_Viewed"] + 1)
    )

    # Purchase conversion relative to wishlist activity
    # High ratio = customer converts interest to purchase
    # Low ratio = customer browses without buying
    df["Wishlist_Conversion"] = (
        df["Frequency"] / (df["Wishlist_Adds"] + 1)
    )

    # ── Churn Signal Features ─────────────────────────
    # NOTE: Engagement_Score removed
    # Was: Session_Count × Avg_Session_Duration
    # Correlation with Session_Count = 0.95
    # Mathematically derived — adds no new information

    # Return Rate
    # Normalises return count by purchase frequency
    # Customer returning 5/5 orders vs 5/50 orders
    # represents very different satisfaction levels
    df["Return_Rate"] = (
        df["Returns"] / (df["Frequency"] + 1)
    )

    # Revenue per Session
    # Captures spending quality per visit
    # NOTE: Dropped from LTV features due to
    # indirect leakage — Revenue_per_Session ×
    # Session_Count reconstructs Monetary (corr 1.0)
    # Retained here for churn model evaluation
    # Results showed no meaningful improvement
    df["Revenue_per_Session"] = (
        df["Monetary"] / (df["Session_Count"] + 1)
    )

    # Abandon Intensity
    # Combines abandon rate with session volume
    # High rate × many sessions = genuinely
    # frustrated customer not just casual browser
    df["Abandon_Intensity"] = (
        df["Cart_Abandon_Rate"] * df["Session_Count"]
    ) / 100

    # ── LTV Target ────────────────────────────────────
    # Used as regression target in LTV model only
    # Monetary weighted by purchase frequency habit
    # log1p smooths extreme frequency outliers
    # More frequent buyers have higher future value
    # Formula: LTV = Monetary × log(Frequency + 1)
    df["LTV"] = df["Monetary"] * np.log1p(df["Frequency"])

    # ── Churn Label ───────────────────────────────────
    # Binary classification target
    # Recency > 90 days = churned customer
    # Recency column dropped in modeling scripts
    # to prevent data leakage
    df["Churn"] = df["Recency"].apply(
        lambda x: 1 if x > 90 else 0
    )

    return df