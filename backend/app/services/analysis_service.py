import numpy as np
import pandas as pd
from app.services.data_service import fetch_stock_data


def calculate_rsi(series: pd.Series, window: int = 14):
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]


def analyze_stock(symbol: str):

    df = fetch_stock_data(symbol)

    if df is None or df.empty:
        return {"error": "No data found for this ticker."}

    df["returns"] = df["Close"].pct_change()

    volatility = df["returns"].std() * np.sqrt(252)
    sharpe_ratio = df["returns"].mean() / df["returns"].std() * np.sqrt(252)
    max_drawdown = (df["Close"] / df["Close"].cummax() - 1).min()

    rsi_value = calculate_rsi(df["Close"])

    trend = "Bullish" if df["Close"].iloc[-1] > df["Close"].mean() else "Bearish"

    return {
        "trend": trend,
        "rating": "Quant-Based",
        "metrics": {
            "rsi": round(float(rsi_value), 2),
            "volatility": round(float(volatility), 4),
            "sharpe_ratio": round(float(sharpe_ratio), 4),
            "max_drawdown": round(float(max_drawdown), 4),
        }
    }