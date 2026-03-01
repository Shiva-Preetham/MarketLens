import numpy as np
import pandas as pd
from scipy.optimize import minimize
from app.services.data_service import fetch_stock_data


# =========================
# MAX SHARPE PORTFOLIO
# =========================
def optimize_portfolio(tickers, risk_free_rate=0.02):
    price_data = pd.DataFrame()

    for ticker in tickers:
        df = fetch_stock_data(ticker)
        price_data[ticker] = df["Close"]

    returns = price_data.pct_change().dropna()

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    num_assets = len(tickers)

    def portfolio_performance(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        )
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.05, 0.5) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets])

    result = minimize(
        portfolio_performance,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    portfolio_return = np.dot(optimal_weights, mean_returns)
    portfolio_volatility = np.sqrt(
        np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
    )
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    return {
        "type": "Max Sharpe Portfolio",
        "weights": dict(zip(tickers, optimal_weights.round(4))),
        "expected_return": round(portfolio_return, 4),
        "volatility": round(portfolio_volatility, 4),
        "sharpe_ratio": round(sharpe_ratio, 4)
    }


# =========================
# MINIMUM VARIANCE PORTFOLIO
# =========================
def minimum_variance_portfolio(tickers):
    price_data = pd.DataFrame()

    for ticker in tickers:
        df = fetch_stock_data(ticker)
        price_data[ticker] = df["Close"]

    returns = price_data.pct_change().dropna()

    cov_matrix = returns.cov() * 252
    num_assets = len(tickers)

    def portfolio_volatility(weights):
        return np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        )

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.05, 0.5) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets])

    result = minimize(
        portfolio_volatility,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    portfolio_vol = portfolio_volatility(optimal_weights)

    return {
        "type": "Minimum Variance Portfolio",
        "weights": dict(zip(tickers, optimal_weights.round(4))),
        "volatility": round(portfolio_vol, 4)
    }