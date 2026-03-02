import yfinance as yf
import numpy as np
import pandas as pd


def fetch_stock_dataframe(symbol: str, period: str = "1y"):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)

    if df.empty:
        return None

    df["Returns"] = df["Close"].pct_change()
    df = df.dropna()
    return df


def fetch_stock_data(symbol: str, period: str = "1y"):
    df = fetch_stock_dataframe(symbol, period)

    if df is None:
        return {"error": "Invalid symbol or no data found"}

    return df[["Close", "Returns"]].reset_index().to_dict(orient="records")


def calculate_risk_metrics(symbol: str, period: str = "1y"):
    df = fetch_stock_dataframe(symbol, period)

    if df is None:
        return {"error": "Invalid symbol or no data found"}

    returns = df["Returns"].values

    volatility = np.std(returns) * np.sqrt(252)

    risk_free_rate = 0.01
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)

    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_drawdown = np.min(drawdown)

    return {
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown)
    }