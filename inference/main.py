# inference/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

model = None
feature_names = None 


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_names

    if not MODEL_PATH.exists():
        raise RuntimeError(f"model file not found at {MODEL_PATH}")

    print(f"loading model from {MODEL_PATH} ...")
    saved = joblib.load(MODEL_PATH)

    if isinstance(saved, dict):
        model = saved.get("model")
        feature_names = saved.get("feature_names")
    elif isinstance(saved, (list, tuple)) and len(saved) == 2:
        model, feature_names = saved
    else:

        model = saved
        feature_names = None

    if model is None:
        raise RuntimeError(f"Loaded model file but could not parse model object. Type={type(saved)}")

    if feature_names is None:
        raise RuntimeError(
            f"feature_names missing in model.pkl (loaded type={type(saved)}). "
            "Re-train/save model as {'model':..., 'feature_names':...}."
        )

    print(f"model loaded. number of features: {len(feature_names)}")

    model = saved["model"]
    feature_names = saved["feature_names"]
    print(f"model loaded. number of features: {len(feature_names)}")

    yield  

    print("shutting down API...")


app = FastAPI(
    title="Diabetes Prediction API",
    description="FastAPI service for LightGBM diabetes classifier (tabular)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DiabetesFeatures(BaseModel):
    gender: str = Field(..., example="female")
    age: float = Field(..., example=45)
    hypertension: int = Field(..., example=0, description="0 = no, 1 = yes")
    heart_disease: int = Field(..., example=0, description="0 = no, 1 = yes")
    smoking_history: str = Field(..., example="never")
    bmi: float = Field(..., example=27.5)
    HbA1c_level: float = Field(..., example=5.8)
    blood_glucose_level: float = Field(..., example=120)


class PredictionOutput(BaseModel):
    prediction: Literal[0, 1]
    probability_diabetes: float
    probability_no_diabetes: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "n_features": len(feature_names) if feature_names is not None else None,
    }


def build_feature_vector(features: DiabetesFeatures) -> np.ndarray:
    df = pd.DataFrame(
        [
            {
                "gender": features.gender,
                "age": features.age,
                "hypertension": features.hypertension,
                "heart_disease": features.heart_disease,
                "smoking_history": features.smoking_history,
                "bmi": features.bmi,
                "HbA1c_level": features.HbA1c_level,
                "blood_glucose_level": features.blood_glucose_level,
            }
        ]
    )

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)

    return df_aligned.values


@app.post("/predict", response_model=PredictionOutput)
def predict(features: DiabetesFeatures):
    if model is None or feature_names is None:
        raise RuntimeError("Model is not loaded properly")

    X = build_feature_vector(features)

    # Predict class and probabilities
    y_pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]  
        p_no_diabetes = float(proba[0])
        p_diabetes = float(proba[1])
    else:
        p_diabetes = float(y_pred)
        p_no_diabetes = 1.0 - p_diabetes

    return PredictionOutput(
        prediction=int(y_pred),
        probability_diabetes=p_diabetes,
        probability_no_diabetes=p_no_diabetes,
    )
