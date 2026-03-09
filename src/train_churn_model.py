import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from feature_engineering import create_features


def train_churn_model():

    # Load data
    data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "ecommerce_user_segmentation.csv"
    )

    df = pd.read_csv(data_path)

    # Feature engineering
    df = create_features(df)

    # Drop unnecessary columns
    df = df.drop(["Customer_ID", "Segment_Label"], axis=1, errors="ignore")

    # Features & target
    X = df.drop(["Churn", "Recency"], axis=1)
    y = df["Churn"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.pkl")

    joblib.dump(model, model_path)

    return accuracy


if __name__ == "__main__":
    score = train_churn_model()
    print("Churn Model Accuracy:", score)