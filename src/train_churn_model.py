import os
import sys
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (f1_score, roc_auc_score,
                             precision_score, recall_score,
                             accuracy_score)
from lightgbm import LGBMClassifier

sys.path.append(os.path.dirname(__file__))
from feature_engineering import create_features


def train_churn_model():

    # ── Load Data ─────────────────────────────────────
    data_path = os.path.join(
        os.path.dirname(__file__),
        "..", "data",
        "ecommerce_user_segmentation.csv"
    )
    df = pd.read_csv(data_path)

    # ── Feature Engineering ───────────────────────────
    df = create_features(df)

    # ── Drop Columns Not Used In Churn Model ──────────
    drop_cols = [
        "Customer_ID",
        "Segment_Label",
        "Recency",        # used to create Churn label
        "Monetary",       # high correlation with multiple features
        "LTV",            # 1.00 correlation with Monetary
        "Return_Rate",    # engineered feature — made results worse
        "Revenue_per_Session",  # engineered feature — made results worse
        "Abandon_Intensity",    # engineered feature — made results worse
    ]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # ── Features And Target ───────────────────────────
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    print("=" * 60)
    print("   CHURN MODEL TRAINING")
    print("=" * 60)
    print(f"  Total samples  : {X.shape[0]}")
    print(f"  Total features : {X.shape[1]}")
    print(f"  Features       : {X.columns.tolist()}")
    print(f"  Churn rate     : {y.mean():.1%}")

    # ── Train Test Split ──────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    counts = y_train.value_counts()
    print(f"\n  Train size     : {X_train.shape[0]}")
    print(f"  Test size      : {X_test.shape[0]}")
    print(f"  Not Churned    : {counts[0]}")
    print(f"  Churned        : {counts[1]}")

    # ── Model ─────────────────────────────────────────
    # LightGBM selected after comparing RandomForest,
    # XGBoost and LightGBM with 5-fold cross validation
    # Best F1 (0.8406) and Recall (0.9794) across all folds
    model = LGBMClassifier(
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )

    # ── Cross Validation ──────────────────────────────
    print("\n  Running 5-fold cross validation...")
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["f1", "roc_auc", "precision", "recall"]

    cv_scores = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    print("\n" + "=" * 60)
    print("   CROSS VALIDATION RESULTS")
    print("=" * 60)
    print(f"  F1 Score  : {cv_scores['test_f1'].mean():.4f}"
          f" ± {cv_scores['test_f1'].std():.4f}")
    print(f"  ROC-AUC   : {cv_scores['test_roc_auc'].mean():.4f}"
          f" ± {cv_scores['test_roc_auc'].std():.4f}")
    print(f"  Precision : {cv_scores['test_precision'].mean():.4f}"
          f" ± {cv_scores['test_precision'].std():.4f}")
    print(f"  Recall    : {cv_scores['test_recall'].mean():.4f}"
          f" ± {cv_scores['test_recall'].std():.4f}")

    # ── Train Final Model ─────────────────────────────
    model.fit(X_train, y_train)

    # ── Evaluate On Test Set ──────────────────────────
    # Threshold 0.30 selected after F-beta optimization
    # Identical performance to 0.10 and 0.20
    # Most business meaningful threshold
    threshold = 0.30
    y_proba   = model.predict_proba(X_test)[:, 1]
    y_pred    = (y_proba >= threshold).astype(int)

    acc         = accuracy_score(y_test, y_pred)
    f1          = f1_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_proba)
    precision   = precision_score(y_test, y_pred)
    recall      = recall_score(y_test, y_pred)
    false_alarm = 1 - precision
    caught      = int(recall * y_test.sum())

    train_acc = accuracy_score(y_train,
                  model.predict(X_train))
    acc_gap   = train_acc - acc

    print("\n" + "=" * 60)
    print("   TEST SET RESULTS")
    print("=" * 60)
    print(f"  Threshold   : {threshold}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  ROC-AUC     : {auc:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  False Alarm : {false_alarm:.1%}")
    print(f"  ACC Gap     : {acc_gap:.4f}")
    print(f"  Caught      : {caught}/{y_test.sum()}")

    # ── Save Model ────────────────────────────────────
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..", "models",
        "churn_model.pkl"
    )

    joblib.dump(
        {
            "model"    : model,
            "threshold": threshold,
            "features" : X_train.columns.tolist(),
        },
        model_path
    )

    print("\n" + "=" * 60)
    print("   MODEL SAVED ✅")
    print("=" * 60)
    print(f"  Algorithm  : LightGBM")
    print(f"  Threshold  : {threshold}")
    print(f"  Features   : {X_train.shape[1]}")
    print(f"  Saved to   : {model_path}")

    return {
        "f1"         : f1,
        "roc_auc"    : auc,
        "precision"  : precision,
        "recall"     : recall,
        "false_alarm": false_alarm,
        "threshold"  : threshold,
    }


if __name__ == "__main__":
    results = train_churn_model()
