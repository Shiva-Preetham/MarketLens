import yfinance as yf
import pandas as pd
import numpy as np
from app.quant.risk import (
    calculate_daily_returns,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)


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

    daily_returns = calculate_daily_returns(df["Close"])

    volatility = calculate_volatility(daily_returns)
    sharpe_ratio = calculate_sharpe_ratio(daily_returns)
    max_drawdown = calculate_max_drawdown(df["Close"])

    return {
        "volatility": round(float(volatility), 4),
        "sharpe_ratio": round(float(sharpe_ratio), 4),
        "max_drawdown": round(float(max_drawdown), 4),
    }