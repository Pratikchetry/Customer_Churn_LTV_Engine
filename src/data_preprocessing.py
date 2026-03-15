from sklearn.model_selection import train_test_split


def preprocess_data(df):

    # Removing non-numeric columns
    if "Customer_ID" in df.columns:
        df = df.drop("Customer_ID", axis=1)

    if "Segment_Label" in df.columns:
        df = df.drop("Segment_Label", axis=1)

    # Features
    X = df.drop("Churn", axis=1)

    # Target
    y = df["Churn"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test