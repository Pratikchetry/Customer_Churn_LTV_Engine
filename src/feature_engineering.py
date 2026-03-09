import pandas as pd

def create_features(df):

    # Feature 1
    df["Clicks_per_Page"] = df["Clicks"] / (df["Pages_Viewed"] + 1)

    # Feature 2
    df["Wishlist_Conversion"] = df["Frequency"] / (df["Wishlist_Adds"] + 1)

    # Feature 3
    df["Engagement_Score"] = df["Session_Count"] * df["Avg_Session_Duration"]

    # It will create churn label
    df["Churn"] = df["Recency"].apply(lambda x: 1 if x > 90 else 0)

    return df