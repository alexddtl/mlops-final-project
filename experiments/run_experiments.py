import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


DATA_PATH = "data/training/churn_features.csv"


def load_data():

    df = pd.read_csv(DATA_PATH)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def evaluate(model, X_test, y_test):

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    return acc, f1, auc


def main():

    mlflow.set_experiment("churn_experiments")

    X_train, X_test, y_train, y_test = load_data()

    models = {

        "LogisticRegression": LogisticRegression(
            max_iter=1000
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    }

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            print(f"\nTraining {name}...")

            model.fit(X_train, y_train)

            acc, f1, auc = evaluate(
                model,
                X_test,
                y_test
            )

            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("auc", auc)

            mlflow.sklearn.log_model(
                model,
                "model"
            )

            print("Accuracy:", acc)
            print("F1:", f1)
            print("AUC:", auc)


if __name__ == "__main__":
    main()