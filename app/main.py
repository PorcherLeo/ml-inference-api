from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from io import StringIO


from .model_utils import predict_one, load_model
app = FastAPI(title="ML Inference API (Iris)", version="1.0.0")


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0)
    sepal_width: float = Field(..., ge=0)
    petal_length: float = Field(..., ge=0)
    petal_width: float = Field(..., ge=0)




@app.get("/health")
async def health():
    try:
        load_model()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/predict")
async def predict(payload: IrisFeatures):
    features = [
    payload.sepal_length,
    payload.sepal_width,
    payload.petal_length,
    payload.petal_width,
    ]
    pred = predict_one(features)
    return {"prediction": pred}




@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Upload a CSV with columns: sepal_length,sepal_width,petal_length,petal_width
    Returns predictions as a list.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")


    content = (await file.read()).decode("utf-8")
    df = pd.read_csv(StringIO(content))


    required_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    if not all(col in df.columns for col in required_cols):
        raise HTTPException(
        status_code=400,
        detail=f"CSV must contain columns: {', '.join(required_cols)}",
        )


    model = load_model()
    preds = model.predict(df[required_cols])
    return {"predictions": [int(x) for x in preds]}