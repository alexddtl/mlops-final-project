import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

MODEL_PATH = "models"
MODEL_FILE = f"{MODEL_PATH}/churn_model.pkl"


os.makedirs(MODEL_PATH, exist_ok=True)


def load_data():

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    X_train = train.drop("Churn", axis=1)
    y_train = train["Churn"]

    X_test = test.drop("Churn", axis=1)
    y_test = test["Churn"]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model


def evaluate(model, X_test, y_test):

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    return acc, f1, auc


def main():

    mlflow.set_experiment("churn_prediction")

    with mlflow.start_run():

        X_train, X_test, y_train, y_test = load_data()

        model = train_model(X_train, y_train)

        acc, f1, auc = evaluate(model, X_test, y_test)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, MODEL_FILE)

        print("Model saved:", MODEL_FILE)

        print("Metrics:")
        print("Accuracy:", acc)
        print("F1:", f1)
        print("AUC:", auc)


if __name__ == "__main__":
    main()
