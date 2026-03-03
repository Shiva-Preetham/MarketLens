import numpy as np
import pandas as pd
from app.services.data_service import fetch_stock_data
from app.quant.scoring import calculate_composite_score, generate_rating
from app.quant.indicators import calculate_rsi as rsi_series


def analyze_stock(symbol: str):

    df = fetch_stock_data(symbol)

    if df is None or df.empty:
        return {"error": "No data found for this ticker."}

    df["returns"] = df["Close"].pct_change()

    volatility = df["returns"].std() * np.sqrt(252)
    sharpe_ratio = df["returns"].mean() / df["returns"].std() * np.sqrt(252)
    max_drawdown = (df["Close"] / df["Close"].cummax() - 1).min()

    # Use shared indicators module for RSI
    rsi_full = rsi_series(df["Close"])
    rsi_value = rsi_full.iloc[-1]

    trend = "Bullish" if df["Close"].iloc[-1] > df["Close"].mean() else "Bearish"

    composite_score = calculate_composite_score(
        trend=trend,
        rsi=float(rsi_value),
        sharpe=float(sharpe_ratio),
        drawdown=float(max_drawdown),
    )
    rating = generate_rating(composite_score)

    return {
        "trend": trend,
        "rating": rating,
        "metrics": {
            "rsi": round(float(rsi_value), 2),
            "volatility": round(float(volatility), 4),
            "sharpe_ratio": round(float(sharpe_ratio), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "score": composite_score,
        }
    }