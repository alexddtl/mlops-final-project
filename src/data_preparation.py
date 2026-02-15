import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RAW_PATH = "data/raw/churn.csv"

TRAINING_PATH = "data/training"
PROCESSED_PATH = "data/processed"

FEATURES_FILE = "data/training/churn_features.csv"
TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"


os.makedirs(TRAINING_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)


def main():

    # Load raw data
    df = pd.read_csv(RAW_PATH)

    # Drop ID
    df = df.drop("customerID", axis=1)

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove missing values
    df = df.dropna()

    # Encode target
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # One-hot encoding
    cat_cols = df.select_dtypes(include="object").columns

    df = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=True
    )

    # Separate target
    y = df["Churn"]
    X = df.drop("Churn", axis=1)

    # Scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    # Reattach target
    final_df = pd.concat(
        [X_scaled, y.reset_index(drop=True)],
        axis=1
    )

    # ====================================
    # Save full dataset (experiments)
    # ====================================

    final_df.to_csv(
        FEATURES_FILE,
        index=False
    )

    # ====================================
    # Train / Test split
    # ====================================

    train_df, test_df = train_test_split(
        final_df,
        test_size=0.2,
        random_state=42,
        stratify=final_df["Churn"]
    )

    # Save processed data
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    # ====================================
    # Logs
    # ====================================

    print("Datasets generated:")

    print("\nFull dataset:")
    print(FEATURES_FILE)
    print("Shape:", final_df.shape)

    print("\nProcessed data:")
    print(TRAIN_FILE, train_df.shape)
    print(TEST_FILE, test_df.shape)


if __name__ == "__main__":
    main()