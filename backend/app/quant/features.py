"""
Feature engineering for the 5-day trend prediction model.

Covers:
  - Technical indicators  : RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic, CCI
  - Price/volume patterns : returns, gaps, volume ratio, price position
  - Macro / sector proxy  : VIX (^VIX), Nifty 50 (^NSEI) relative strength
  - Sentiment proxy       : news-driven via yfinance .news headline count + polarity
                            (uses VADER if available, else binary title-count heuristic)
"""

import numpy as np
import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(close: pd.Series, period: int = 20):
    sma   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    width = (upper - lower) / sma.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, lower, width, pct_b


def _atr(high, low, close, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _stochastic(high, low, close, k: int = 14, d: int = 3):
    low_k  = low.rolling(k).min()
    high_k = high.rolling(k).max()
    pct_k  = 100 * (close - low_k) / (high_k - low_k).replace(0, np.nan)
    pct_d  = pct_k.rolling(d).mean()
    return pct_k, pct_d


def _cci(high, low, close, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3
    sma     = typical.rolling(period).mean()
    mad     = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (typical - sma) / (0.015 * mad.replace(0, np.nan))


# ─────────────────────────────────────────────
# SENTIMENT PROXY  (headline count + polarity)
# ─────────────────────────────────────────────

def _sentiment_score(ticker_obj) -> float:
    """
    Returns a sentiment score in [-1, 1].
    Uses VADER if installed, otherwise falls back to a simple
    positive/negative keyword count on headlines.
    """
    try:
        news = ticker_obj.news or []
    except Exception:
        return 0.0

    if not news:
        return 0.0

    titles = [n.get("content", {}).get("title", "") or n.get("title", "") for n in news[:20]]
    titles = [t for t in titles if t]

    if not titles:
        return 0.0

    # Try VADER first
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
        sia    = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(t)["compound"] for t in titles]
        return float(np.mean(scores))
    except ImportError:
        pass

    # Keyword fallback
    POS = {"surge", "rally", "gain", "rise", "high", "beat", "strong", "bull", "up", "growth", "profit"}
    NEG = {"fall", "drop", "crash", "loss", "low", "miss", "weak", "bear", "down", "decline", "sell"}
    scores = []
    for t in titles:
        words = set(t.lower().split())
        p = len(words & POS)
        n = len(words & NEG)
        if p + n > 0:
            scores.append((p - n) / (p + n))
        else:
            scores.append(0.0)
    return float(np.mean(scores))


# ─────────────────────────────────────────────
# MACRO FEATURES  (VIX + NIFTY relative)
# ─────────────────────────────────────────────

def _macro_features(symbol: str, period: str, index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Downloads VIX and aligns it with the symbol's trading calendar.
    Returns a DataFrame with columns: vix_close, vix_chg, rel_strength
    """
    macro = pd.DataFrame(index=index_df.index)

    # VIX
    try:
        vix = yf.Ticker("^VIX").history(period=period)[["Close"]].rename(columns={"Close": "vix_close"})
        vix.index = vix.index.tz_localize(None) if vix.index.tz else vix.index
        macro = macro.join(vix, how="left")
        macro["vix_close"] = macro["vix_close"].ffill()
        macro["vix_chg"]   = macro["vix_close"].pct_change()
    except Exception:
        macro["vix_close"] = np.nan
        macro["vix_chg"]   = np.nan

    # NIFTY relative strength (skip if symbol IS nifty)
    if symbol not in ("^NSEI", "NIFTY50"):
        try:
            nifty = yf.Ticker("^NSEI").history(period=period)[["Close"]].rename(columns={"Close": "nifty"})
            nifty.index = nifty.index.tz_localize(None) if nifty.index.tz else nifty.index
            macro = macro.join(nifty, how="left")
            macro["nifty"] = macro["nifty"].ffill()
            macro["rel_strength"] = index_df["Close"].pct_change() - macro["nifty"].pct_change()
            macro.drop(columns=["nifty"], inplace=True)
        except Exception:
            macro["rel_strength"] = np.nan
    else:
        macro["rel_strength"] = 0.0

    return macro


# ─────────────────────────────────────────────
# MASTER FEATURE BUILDER
# ─────────────────────────────────────────────

def build_features(symbol: str, period: str = "2y") -> pd.DataFrame:
    """
    Downloads OHLCV for *symbol* and returns a fully-featured DataFrame.
    The TARGET column is: 1 if Close[t+5] > Close[t] else 0  (5-day direction).
    Drops rows with any NaN after feature construction.
    """
    ticker = yf.Ticker(symbol)
    df     = ticker.history(period=period)

    if df.empty or len(df) < 60:
        raise ValueError(f"Insufficient data for {symbol}")

    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # ── Technical ──
    df["rsi_14"]        = _rsi(c, 14)
    df["rsi_7"]         = _rsi(c, 7)
    macd, sig, hist     = _macd(c)
    df["macd"]          = macd
    df["macd_signal"]   = sig
    df["macd_hist"]     = hist
    _, _, bb_width, pct_b = _bollinger(c, 20)
    df["bb_width"]      = bb_width
    df["bb_pct_b"]      = pct_b
    df["atr_14"]        = _atr(h, l, c, 14)
    df["atr_pct"]       = df["atr_14"] / c          # normalised ATR
    df["obv"]           = _obv(c, v)
    df["obv_chg"]       = df["obv"].pct_change(5)
    pct_k, pct_d        = _stochastic(h, l, c)
    df["stoch_k"]       = pct_k
    df["stoch_d"]       = pct_d
    df["cci_20"]        = _cci(h, l, c, 20)

    # ── Price / Volume patterns ──
    df["ret_1"]         = c.pct_change(1)
    df["ret_3"]         = c.pct_change(3)
    df["ret_5"]         = c.pct_change(5)
    df["ret_10"]        = c.pct_change(10)
    df["ret_20"]        = c.pct_change(20)
    df["gap"]           = (df["Open"] - c.shift(1)) / c.shift(1)
    df["hl_range"]      = (h - l) / c              # daily range %
    df["close_pos"]     = (c - l) / (h - l).replace(0, np.nan)  # where close sits in day range
    df["vol_ratio_5"]   = v / v.rolling(5).mean().replace(0, np.nan)
    df["vol_ratio_20"]  = v / v.rolling(20).mean().replace(0, np.nan)
    df["price_vs_sma20"]= c / c.rolling(20).mean().replace(0, np.nan) - 1
    df["price_vs_sma50"]= c / c.rolling(50).mean().replace(0, np.nan) - 1
    df["rolling_std_10"]= c.pct_change().rolling(10).std()

    # ── Macro ──
    macro = _macro_features(symbol, period, df)
    df = df.join(macro, how="left")

    # ── Sentiment (single scalar added as constant column, refreshed at predict time) ──
    df["sentiment"] = _sentiment_score(ticker)

    # ── Target: 1 if price is higher 5 days later ──
    df["target"] = (c.shift(-5) > c).astype(int)

    # Drop last 5 rows (no future label) and all NaN rows
    df = df.iloc[:-5].dropna()

    return df


def feature_columns(df: pd.DataFrame) -> list:
    """Return all feature column names (everything except OHLCV and target)."""
    skip = {"Open", "High", "Low", "Close", "Volume", "target"}
    return [c for c in df.columns if c not in skip]
