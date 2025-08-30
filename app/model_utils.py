from __future__ import annotations
import os
import joblib
import numpy as np
from typing import List


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
MODEL_PATH = os.path.abspath(MODEL_PATH)


_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train it with train/train.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_one(features: List[float]) -> int:
    model = load_model()
    X = np.array(features, dtype=float).reshape(1, -1)
    y_pred = model.predict(X)
    return int(y_pred[0])