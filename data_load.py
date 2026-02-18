import yfinance as yf

def get_stock_data(ticker="RELIANCE.NS", period="1y"):
    data = yf.download(ticker, period=period)
    print(data.head())
    return data

if __name__ == "__main__":
    get_stock_data()
