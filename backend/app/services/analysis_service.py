from app.services.data_service import fetch_stock_data
from app.quant.indicators import moving_average, calculate_rsi
from app.quant.risk import (
    calculate_daily_returns,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)
from app.quant.scoring import calculate_composite_score, generate_rating


def analyze_stock(ticker: str):
    df = fetch_stock_data(ticker)

    if df.empty:
        return {"error": "Invalid ticker or no data available"}

    df["MA_50"] = moving_average(df["Close"], 50)
    df["MA_200"] = moving_average(df["Close"], 200)
    df["RSI"] = calculate_rsi(df["Close"])

    daily_returns = calculate_daily_returns(df["Close"])
    volatility = calculate_volatility(daily_returns)
    sharpe = calculate_sharpe_ratio(daily_returns)
    max_drawdown = calculate_max_drawdown(df["Close"])

    latest = df.iloc[-1]

    trend = "Bullish" if latest["MA_50"] > latest["MA_200"] else "Bearish"

    score = calculate_composite_score(
        trend,
        latest["RSI"],
        sharpe,
        max_drawdown
    )

    rating = generate_rating(score)

    return {
        "ticker": ticker.upper(),
        "metrics": {
            "ma_50": round(latest["MA_50"], 2),
            "ma_200": round(latest["MA_200"], 2),
            "rsi": round(latest["RSI"], 2),
            "volatility": round(volatility, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 4),
        },
        "trend": trend,
        "score": score,
        "rating": rating,
    }