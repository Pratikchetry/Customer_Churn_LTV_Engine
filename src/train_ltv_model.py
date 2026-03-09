import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from feature_engineering import create_features


def train_ltv_model():

    # Correct path to dataset
    data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "ecommerce_user_segmentation.csv"
    )

    df = pd.read_csv(data_path)

    # Apply feature engineering
    df = create_features(df)

    # Remove unnecessary columns
    df = df.drop(["Customer_ID", "Segment_Label"], axis=1)

    # Features and target
    X = df.drop("Monetary", axis=1)
    y = df["Monetary"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Model
    model = XGBRegressor()

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # R2 Score
    r2 = r2_score(y_test, y_pred)

    # Save model correctly
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "ltv_model.pkl")
    
    joblib.dump(model, model_path)

    return r2


if __name__ == "__main__":

    score = train_ltv_model()

    print("LTV Model R2 Score:", score)