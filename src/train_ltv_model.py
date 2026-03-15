import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.append(os.path.dirname(__file__))
from feature_engineering import create_features


def train_ltv_model():

    # ── Load Data ─────────────────────────────────────
    data_path = os.path.join(
        os.path.dirname(__file__),
        "..", "data",
        "ecommerce_user_segmentation.csv"
    )
    df = pd.read_csv(data_path)

    # ── Feature Engineering ───────────────────────────
    df = create_features(df)

    # ── Drop Columns Not Used In LTV Model ────────────
    # Monetary dropped → used to create LTV target (leakage)
    # LTV dropped → this is our target
    # Revenue_per_Session dropped → indirect leakage
    #   Revenue_per_Session × Session_Count = Monetary (corr 1.0)
    # Engagement_Score dropped → 0.95 corr with Session_Count
    # Return_Rate, Abandon_Intensity → no meaningful improvement
    drop_cols = [
        "Customer_ID",
        "Segment_Label",
        "Recency",
        "Churn",
        "Monetary",
        "LTV",
        "Return_Rate",
        "Revenue_per_Session",
        "Abandon_Intensity",
    ]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # ── Features And Target ───────────────────────────
    X = df
    y = create_features(
            pd.read_csv(data_path)
        )["LTV"]

    print("=" * 60)
    print("   LTV MODEL TRAINING")
    print("=" * 60)
    print(f"  Total samples  : {X.shape[0]}")
    print(f"  Total features : {X.shape[1]}")
    print(f"  Features       : {X.columns.tolist()}")
    print(f"  Target         : LTV")
    print(f"  LTV Mean       : ${y.mean():,.2f}")
    print(f"  LTV Std        : ${y.std():,.2f}")
    print(f"  LTV Skewness   : {y.skew():.4f}")

    # ── Train Test Split ──────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    print(f"\n  Train size     : {X_train.shape[0]}")
    print(f"  Test size      : {X_test.shape[0]}")

    # ── Model ─────────────────────────────────────────
    # RandomForest selected after comparing with
    # XGBoost and LightGBM across multiple configurations:
    # baseline, extended features, regularization,
    # and log transform experiments
    # Best R² (0.9226), lowest RMSE ($2,806)
    # Only model with R² gap below 0.05 threshold (0.0463)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=0.7,
        random_state=42,
        n_jobs=-1
    )

    # ── Cross Validation ──────────────────────────────
    print("\n  Running 5-fold cross validation...")
    cv      = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["r2", "neg_root_mean_squared_error",
               "neg_mean_absolute_error"]

    cv_scores = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    cv_r2   = cv_scores["test_r2"].mean()
    cv_rmse = (-cv_scores[
        "test_neg_root_mean_squared_error"
    ]).mean()
    cv_mae  = (-cv_scores[
        "test_neg_mean_absolute_error"
    ]).mean()

    print("\n" + "=" * 60)
    print("   CROSS VALIDATION RESULTS")
    print("=" * 60)
    print(f"  R²    : {cv_r2:.4f}"
          f" ± {cv_scores['test_r2'].std():.4f}")
    print(f"  RMSE  : ${cv_rmse:,.2f}"
          f" ± ${(-cv_scores['test_neg_root_mean_squared_error']).std():,.2f}")
    print(f"  MAE   : ${cv_mae:,.2f}")

    # ── Train Final Model ─────────────────────────────
    model.fit(X_train, y_train)

    # ── Evaluate On Test Set ──────────────────────────
    y_pred     = model.predict(X_test)
    r2         = r2_score(y_test, y_pred)
    rmse       = np.sqrt(mean_squared_error(y_test, y_pred))
    mae        = mean_absolute_error(y_test, y_pred)
    error_rate = rmse / y.mean()

    # Overfitting check
    y_train_pred = model.predict(X_train)
    r2_train     = r2_score(y_train, y_train_pred)
    r2_gap       = r2_train - r2

    print("\n" + "=" * 60)
    print("   TEST SET RESULTS")
    print("=" * 60)
    print(f"  R²          : {r2:.4f}")
    print(f"  RMSE        : ${rmse:,.2f}")
    print(f"  MAE         : ${mae:,.2f}")
    print(f"  Error Rate  : {error_rate:.1%}")
    print(f"  R² Train    : {r2_train:.4f}")
    print(f"  R² Gap      : {r2_gap:.4f}"
          f"  {'✅' if r2_gap < 0.05 else '⚠️'}")

    # ── Save Model ────────────────────────────────────
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..", "models",
        "ltv_model.pkl"
    )

    joblib.dump(
        {
            "model"   : model,
            "features": X_train.columns.tolist(),
            "target"  : "LTV",
            "transform": "none",
        },
        model_path
    )

    print("\n" + "=" * 60)
    print("   MODEL SAVED ✅")
    print("=" * 60)
    print(f"  Algorithm  : RandomForest")
    print(f"  Features   : {X_train.shape[1]}")
    print(f"  R²         : {r2:.4f}")
    print(f"  RMSE       : ${rmse:,.2f}")
    print(f"  R² Gap     : {r2_gap:.4f}")
    print(f"  Saved to   : {model_path}")

    return {
        "r2"        : r2,
        "rmse"      : rmse,
        "mae"       : mae,
        "error_rate": error_rate,
        "r2_gap"    : r2_gap,
    }


if __name__ == "__main__":
    results = train_ltv_model()
