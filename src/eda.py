import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
RAW_DATA_PATH = "data/raw/churn.csv"
REPORT_PATH = "reports/eda"

os.makedirs(REPORT_PATH, exist_ok=True)


def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def basic_info(df):
    print("\n=== INFO ===")
    print(df.info())

    print("\n=== DESCRIBE ===")
    print(df.describe())

    print("\n=== NULL VALUES ===")
    print(df.isnull().sum())


def plot_target(df):

    plt.figure()
    sns.countplot(x="Churn", data=df)
    plt.title("Churn Distribution")

    path = f"{REPORT_PATH}/churn_distribution.png"
    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")


def plot_numeric(df):

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:

        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(col)

        path = f"{REPORT_PATH}/{col}_hist.png"
        plt.savefig(path)
        plt.close()

        print(f"Saved: {path}")


def plot_correlations(df):

    plt.figure(figsize=(10, 8))

    corr = df.corr()

    sns.heatmap(corr, cmap="coolwarm")

    path = f"{REPORT_PATH}/correlation.png"
    plt.savefig(path)
    plt.close()

    print(f"Saved: {path}")


def main():

    df = load_data()

    basic_info(df)

    plot_target(df)

    plot_numeric(df)

    plot_correlations(df)


if __name__ == "__main__":
    main()
