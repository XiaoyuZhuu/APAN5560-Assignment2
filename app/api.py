# app/api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from .inference import Predictor

app = FastAPI(title="Assignment 2 Image Classifier")
_predictor = None

def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor("models/cnn_cifar10.pt")  
    return _predictor

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pred = _get_predictor().predict_bytes(image_bytes)
    return JSONResponse(pred)
