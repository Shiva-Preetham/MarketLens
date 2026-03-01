import yfinance as yf


def fetch_stock_data(ticker: str, period: str = "1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df