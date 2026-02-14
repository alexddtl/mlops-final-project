import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = "models/churn_model.pkl"

model = joblib.load(MODEL_PATH)

app = FastAPI()


class Customer(BaseModel):

    features: list


@app.post("/predict")
def predict(data: Customer):

    X = np.array(data.features).reshape(1, -1)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return {
        "churn_prediction": int(pred),
        "churn_probability": float(prob)
    }