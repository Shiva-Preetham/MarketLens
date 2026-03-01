import pandas as pd
import numpy as np


def moving_average(data: pd.Series, window: int):
    return data.rolling(window=window).mean()


def calculate_rsi(data: pd.Series, window: int = 14):
    delta = data.diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi