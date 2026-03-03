"""
ML service — thin async-friendly wrapper around app.quant.ml_model.
Called by the /predict router. Handles errors gracefully.
"""

from dataclasses import asdict
from app.quant.ml_model import train, predict, model_info


def train_model(symbol: str) -> dict:
    """Train (or retrain) the ensemble model for a symbol. Returns metrics dict."""
    try:
        result = train(symbol)
        return {"status": "ok", **asdict(result)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def predict_signal(symbol: str) -> dict:
    """
    Return 5-day directional prediction.
    Auto-trains if no cached model exists.
    """
    try:
        result = predict(symbol, retrain_if_missing=True)
        return {"status": "ok", **asdict(result)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def get_model_info(symbol: str) -> dict:
    """Return cached model metadata without retraining."""
    info = model_info(symbol)
    if info is None:
        return {"status": "not_trained"}
    return {"status": "ok", **info}
