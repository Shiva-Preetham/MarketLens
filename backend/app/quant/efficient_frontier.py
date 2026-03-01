import numpy as np
import pandas as pd
from app.services.data_service import fetch_stock_data


def generate_efficient_frontier(tickers, num_portfolios=500, risk_free_rate=0.02):
    price_data = pd.DataFrame()

    for ticker in tickers:
        df = fetch_stock_data(ticker)
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        price_data[ticker] = df["Close"]

    returns = price_data.pct_change().dropna()

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    results = []

    for _ in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        )

        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results.append({
            "return": portfolio_return,
            "volatility": portfolio_volatility,
            "sharpe_ratio": sharpe_ratio,
            "weights": dict(zip(tickers, weights.round(4)))
        })

    return results