import yfinance as yf
import pandas as pd
import numpy as np


# =========================
# FETCH STOCK DATA
# =========================
def fetch_stock_data(symbol: str, period: str = "1y"):

    stock = yf.Ticker(symbol)
    df = stock.history(period=period)

    if df.empty:
        return None

    df.reset_index(inplace=True)
    return df


# =========================
# RISK METRICS
# =========================
def calculate_risk_metrics(symbol: str, period: str = "1y"):

    df = fetch_stock_data(symbol, period)

    if df is None or df.empty:
        return {"error": "No data found."}

    df["returns"] = df["Close"].pct_change()

    volatility = df["returns"].std() * np.sqrt(252)
    sharpe_ratio = df["returns"].mean() / df["returns"].std() * np.sqrt(252)
    max_drawdown = (df["Close"] / df["Close"].cummax() - 1).min()

    return {
        "volatility": round(float(volatility), 4),
        "sharpe_ratio": round(float(sharpe_ratio), 4),
        "max_drawdown": round(float(max_drawdown), 4),
    }