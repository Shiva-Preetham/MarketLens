"""
/predict router

GET  /predict/{ticker}         → predict 5-day direction (auto-trains if needed)
POST /predict/{ticker}/train   → force retrain model
GET  /predict/{ticker}/info    → cached model metrics (no retrain)
"""

from fastapi import APIRouter, BackgroundTasks
from app.services.ml_service import predict_signal, train_model, get_model_info

router = APIRouter()


@router.get("/predict/{ticker}")
def get_prediction(ticker: str):
    """
    Predict the 5-day market direction for a ticker.
    Returns signal (BULLISH/BEARISH), confidence, probabilities,
    sentiment score, top feature snapshot and model accuracy.
    First call will trigger training (~30s). Subsequent calls use cache.
    """
    return predict_signal(ticker)


@router.post("/predict/{ticker}/train")
def retrain_model(ticker: str, background_tasks: BackgroundTasks):
    """
    Force a full model retrain for a ticker.
    Training runs in the background; returns immediately with status.
    """
    background_tasks.add_task(train_model, ticker)
    return {"status": "training_started", "symbol": ticker,
            "message": "Model training started in background. Check /predict/{ticker}/info for progress."}


@router.get("/predict/{ticker}/info")
def model_metadata(ticker: str):
    """
    Return cached model metrics (accuracy, AUC, feature importances, etc.)
    without triggering a retrain.
    """
    return get_model_info(ticker)
