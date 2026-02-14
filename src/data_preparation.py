import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RAW_PATH = "data/raw/churn.csv"
PROCESSED_PATH = "data/processed"
TRAIN_PATH = f"{PROCESSED_PATH}/train.csv"
TEST_PATH = f"{PROCESSED_PATH}/test.csv"


os.makedirs(PROCESSED_PATH, exist_ok=True)


def load_data():
    return pd.read_csv(RAW_PATH)


def clean_data(df):

    df = df.drop("customerID", axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna()

    return df


def encode_data(df):

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    cat_cols = df.select_dtypes(include="object").columns

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


def split_scale(df):

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train = pd.concat(
        [pd.DataFrame(X_train), y_train.reset_index(drop=True)],
        axis=1
    )

    test = pd.concat(
        [pd.DataFrame(X_test), y_test.reset_index(drop=True)],
        axis=1
    )

    return train, test


def save_data(train, test):

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print("Saved datasets:")
    print(TRAIN_PATH)
    print(TEST_PATH)


def main():

    df = load_data()

    df = clean_data(df)

    df = encode_data(df)

    train, test = split_scale(df)

    save_data(train, test)


if __name__ == "__main__":
    main()
