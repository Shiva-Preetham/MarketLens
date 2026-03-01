import numpy as np


def calculate_daily_returns(close_prices):
    return close_prices.pct_change()


def calculate_volatility(daily_returns):
    return daily_returns.std() * np.sqrt(252)


def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.02):
    excess_returns = daily_returns - (risk_free_rate / 252)
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)


def calculate_max_drawdown(close_prices):
    cumulative_max = close_prices.cummax()
    drawdown = (close_prices - cumulative_max) / cumulative_max
    return drawdown.min()