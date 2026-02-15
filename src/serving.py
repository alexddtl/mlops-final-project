import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict


MODEL_PATH = "models/churn_model.pkl"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
FEATURES = bundle["features"]
MEDIANS = bundle["medians"]


app = FastAPI(title="Churn Prediction API")


class Customer(BaseModel):
    data: Dict[str, float]


@app.post("/predict")
def predict(request: Customer):

    input_data = request.data

    # Create base input using medians
    row = {}

    for col in FEATURES:
        row[col] = MEDIANS[col]

    # Override with provided values
    for col, val in input_data.items():

        if col not in FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown feature: {col}"
            )

        row[col] = val

    df = pd.DataFrame([row])

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(pred),
        "churn_probability": float(prob)
    }