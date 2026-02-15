import pandas as pd
import os


RAW_PATH = "data/raw/churn.csv"
OUT_PATH = "data/training/churn_features.csv"


def main():

    print("Loading raw data...")

    df = pd.read_csv(RAW_PATH)

    # Clean TotalCharges (sometimes string)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Drop missing
    df = df.dropna()

    # Drop ID
    df = df.drop("customerID", axis=1)

    # One-hot encoding
    df = pd.get_dummies(
        df,
        drop_first=True
    )

    # Create folder
    os.makedirs("data/training", exist_ok=True)

    # Save
    df.to_csv(OUT_PATH, index=False)

    print("Features saved to:")
    print(OUT_PATH)

    print("Shape:", df.shape)


if __name__ == "__main__":
    main()