# MLOps Introduction: Final Project
FInal work description in  the [final_project_description.md](final_project_description.md) file.

Student info:
- Full name: Alexander David Tapia Lescano
- e-mail: alexddtl@gmail.com


## Project Name: [Churn Prediction - MLOps Final Project]

## Overview

This project implements an end-to-end MLOps pipeline for customer churn prediction,
including data preparation, model training, experiment tracking, deployment and serving.

## Project Structure

mlops-final-project/
│
├── data/
│ ├── raw/ # Original dataset
│ ├── processed/ # Intermediate datasets
│ └── training/ # Final training dataset
│
├── experiments/ # Model experimentation scripts
│
├── models/ # Trained models
│
├── notebooks/ # Optional exploratory notebooks
│
├── reports/ # EDA plots and experiment results
│
├── src/
│ ├── eda.py
│ ├── data_preparation.py
│ ├── train.py
│ ├── build_features.py
│ └── serving.py
│
├── tests/
│
├── requirements.txt
└── README.md

## Data
https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download

Telco Customer Churn Dataset.

Target variable: `Churn` (Yes/No)

## Pipeline

### 1. Exploratory Data Analysis

python src/eda.py

Generates plots in `reports/eda/`.

### 2. Data Preparation

python src/data_preparation.py

Outputs:

Full dataset:
data/training/churn_features.csv
Shape: (7032, 31)

Processed data:
data/processed/train.csv (5625, 31)
data/processed/test.csv (1407, 31)

python src/build_features.py

### 3. Model Experimentation

python experiments/run_experiments.py

Models evaluated:

Logistic Regression

Random Forest

Metrics:

Accuracy

F1-score

ROC-AUC

Tracked using MLflow.

Results:

Model	ROC-AUC
Logistic Regression	0.83
Random Forest	0.81



### 4. Model Training

python src/train.py

- Trains Logistic Regression
- Logs metrics with MLflow
- Saves model to `models/`

Output:

models/churn_model.pkl

### 5. Experiment Tracking
Access at: http://127.0.0.1:5000

### 6. Model Serving
uvicorn src.serving:app --reload

API available at: http://127.0.0.1:8000

### 7. Prediction Example


API available at: http://127.0.0.1:8000

PowerShell:

```powershell
$body = @{
  data = @{
    tenure = 12
    MonthlyCharges = 70
    TotalCharges = 840
    gender_Male = 1
  }
} | ConvertTo-Json

Invoke-RestMethod `
 -Uri "http://127.0.0.1:8000/predict" `
 -Method Post `
 -ContentType "application/json" `
 -Body $body


## MLOps Stack

- Python
- Scikit-Learn
- MLflow
- FastAPI
- DVC
- GitHub

## Reproducibility

### Setup

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

### Data Versioning

dvc pull

### Reproduce Pipeline

dvc repro

### Training 
python src/train.py

### Serving

uvicorn src.serving:app --reload



