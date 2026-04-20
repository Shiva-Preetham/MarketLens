# """
# Feature engineering for the 5-day trend prediction model.

# Covers:
#   - Technical indicators  : RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic, CCI
#   - Price/volume patterns : returns, gaps, volume ratio, price position
#   - Macro / sector proxy  : VIX (^VIX), Nifty 50 (^NSEI) relative strength
#   - Sentiment proxy       : news-driven via yfinance .news headline count + polarity
#                             (uses VADER if available, else binary title-count heuristic)
# """

# import numpy as np
# import pandas as pd
# import yfinance as yf


# # ─────────────────────────────────────────────
# # TECHNICAL INDICATORS
# # ─────────────────────────────────────────────

# def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
#     delta = close.diff()
#     gain  = delta.clip(lower=0).rolling(period).mean()
#     loss  = (-delta.clip(upper=0)).rolling(period).mean()
#     rs    = gain / loss.replace(0, np.nan)
#     return 100 - (100 / (1 + rs))


# def _macd(close: pd.Series):
#     ema12 = close.ewm(span=12, adjust=False).mean()
#     ema26 = close.ewm(span=26, adjust=False).mean()
#     macd_line   = ema12 - ema26
#     signal_line = macd_line.ewm(span=9, adjust=False).mean()
#     histogram   = macd_line - signal_line
#     return macd_line, signal_line, histogram


# def _bollinger(close: pd.Series, period: int = 20):
#     sma   = close.rolling(period).mean()
#     std   = close.rolling(period).std()
#     upper = sma + 2 * std
#     lower = sma - 2 * std
#     width = (upper - lower) / sma.replace(0, np.nan)
#     pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
#     return upper, lower, width, pct_b


# def _atr(high, low, close, period: int = 14) -> pd.Series:
#     tr = pd.concat([
#         high - low,
#         (high - close.shift()).abs(),
#         (low  - close.shift()).abs()
#     ], axis=1).max(axis=1)
#     return tr.rolling(period).mean()


# def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
#     direction = np.sign(close.diff()).fillna(0)
#     return (direction * volume).cumsum()


# def _stochastic(high, low, close, k: int = 14, d: int = 3):
#     low_k  = low.rolling(k).min()
#     high_k = high.rolling(k).max()
#     pct_k  = 100 * (close - low_k) / (high_k - low_k).replace(0, np.nan)
#     pct_d  = pct_k.rolling(d).mean()
#     return pct_k, pct_d


# def _cci(high, low, close, period: int = 20) -> pd.Series:
#     typical = (high + low + close) / 3
#     sma     = typical.rolling(period).mean()
#     mad     = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
#     return (typical - sma) / (0.015 * mad.replace(0, np.nan))


# # ─────────────────────────────────────────────
# # SENTIMENT PROXY  (headline count + polarity)
# # ─────────────────────────────────────────────

# def _sentiment_score(ticker_obj) -> float:
#     """
#     Returns a sentiment score in [-1, 1].
#     Uses VADER if installed, otherwise falls back to a simple
#     positive/negative keyword count on headlines.
#     """
#     try:
#         news = ticker_obj.news or []
#     except Exception:
#         return 0.0

#     if not news:
#         return 0.0

#     titles = [n.get("content", {}).get("title", "") or n.get("title", "") for n in news[:20]]
#     titles = [t for t in titles if t]

#     if not titles:
#         return 0.0

#     # Try VADER first
#     try:
#         from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
#         sia    = SentimentIntensityAnalyzer()
#         scores = [sia.polarity_scores(t)["compound"] for t in titles]
#         return float(np.mean(scores))
#     except ImportError:
#         pass

#     # Keyword fallback
#     POS = {"surge", "rally", "gain", "rise", "high", "beat", "strong", "bull", "up", "growth", "profit"}
#     NEG = {"fall", "drop", "crash", "loss", "low", "miss", "weak", "bear", "down", "decline", "sell"}
#     scores = []
#     for t in titles:
#         words = set(t.lower().split())
#         p = len(words & POS)
#         n = len(words & NEG)
#         if p + n > 0:
#             scores.append((p - n) / (p + n))
#         else:
#             scores.append(0.0)
#     return float(np.mean(scores))


# # ─────────────────────────────────────────────
# # MACRO FEATURES  (VIX + NIFTY relative)
# # ─────────────────────────────────────────────

# def _macro_features(symbol: str, period: str, index_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Downloads VIX and aligns it with the symbol's trading calendar.
#     Returns a DataFrame with columns: vix_close, vix_chg, rel_strength
#     """
#     macro = pd.DataFrame(index=index_df.index)

#     # VIX
#     try:
#         vix = yf.Ticker("^VIX").history(period=period)[["Close"]].rename(columns={"Close": "vix_close"})
#         vix.index = vix.index.tz_localize(None) if vix.index.tz else vix.index
#         macro = macro.join(vix, how="left")
#         macro["vix_close"] = macro["vix_close"].ffill()
#         macro["vix_chg"]   = macro["vix_close"].pct_change()
#     except Exception:
#         macro["vix_close"] = np.nan
#         macro["vix_chg"]   = np.nan

#     # NIFTY relative strength (skip if symbol IS nifty)
#     if symbol not in ("^NSEI", "NIFTY50"):
#         try:
#             nifty = yf.Ticker("^NSEI").history(period=period)[["Close"]].rename(columns={"Close": "nifty"})
#             nifty.index = nifty.index.tz_localize(None) if nifty.index.tz else nifty.index
#             macro = macro.join(nifty, how="left")
#             macro["nifty"] = macro["nifty"].ffill()
#             macro["rel_strength"] = index_df["Close"].pct_change() - macro["nifty"].pct_change()
#             macro.drop(columns=["nifty"], inplace=True)
#         except Exception:
#             macro["rel_strength"] = np.nan
#     else:
#         macro["rel_strength"] = 0.0

#     return macro


# # ─────────────────────────────────────────────
# # MASTER FEATURE BUILDER
# # ─────────────────────────────────────────────

# def build_features(symbol: str, period: str = "2y") -> pd.DataFrame:
#     """
#     Downloads OHLCV for *symbol* and returns a fully-featured DataFrame.
#     The TARGET column is: 1 if Close[t+5] > Close[t] else 0  (5-day direction).
#     Drops rows with any NaN after feature construction.
#     """
#     ticker = yf.Ticker(symbol)
#     df     = ticker.history(period=period)

#     if df.empty or len(df) < 60:
#         raise ValueError(f"Insufficient data for {symbol}")

#     df.index = df.index.tz_localize(None) if df.index.tz else df.index
#     df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

#     c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

#     # ── Technical ──
#     df["rsi_14"]        = _rsi(c, 14)
#     df["rsi_7"]         = _rsi(c, 7)
#     macd, sig, hist     = _macd(c)
#     df["macd"]          = macd
#     df["macd_signal"]   = sig
#     df["macd_hist"]     = hist
#     _, _, bb_width, pct_b = _bollinger(c, 20)
#     df["bb_width"]      = bb_width
#     df["bb_pct_b"]      = pct_b
#     df["atr_14"]        = _atr(h, l, c, 14)
#     df["atr_pct"]       = df["atr_14"] / c          # normalised ATR
#     df["obv"]           = _obv(c, v)
#     df["obv_chg"]       = df["obv"].pct_change(5)
#     pct_k, pct_d        = _stochastic(h, l, c)
#     df["stoch_k"]       = pct_k
#     df["stoch_d"]       = pct_d
#     df["cci_20"]        = _cci(h, l, c, 20)

#     # ── Price / Volume patterns ──
#     df["ret_1"]         = c.pct_change(1)
#     df["ret_3"]         = c.pct_change(3)
#     df["ret_5"]         = c.pct_change(5)
#     df["ret_10"]        = c.pct_change(10)
#     df["ret_20"]        = c.pct_change(20)
#     df["gap"]           = (df["Open"] - c.shift(1)) / c.shift(1)
#     df["hl_range"]      = (h - l) / c              # daily range %
#     df["close_pos"]     = (c - l) / (h - l).replace(0, np.nan)  # where close sits in day range
#     df["vol_ratio_5"]   = v / v.rolling(5).mean().replace(0, np.nan)
#     df["vol_ratio_20"]  = v / v.rolling(20).mean().replace(0, np.nan)
#     df["price_vs_sma20"]= c / c.rolling(20).mean().replace(0, np.nan) - 1
#     df["price_vs_sma50"]= c / c.rolling(50).mean().replace(0, np.nan) - 1
#     df["rolling_std_10"]= c.pct_change().rolling(10).std()

#     # ── Macro ──
#     macro = _macro_features(symbol, period, df)
#     df = df.join(macro, how="left")

#     # ── Sentiment (single scalar added as constant column, refreshed at predict time) ──
#     df["sentiment"] = _sentiment_score(ticker)

#     # ── Target: 1 if price is higher 5 days later ──
#     df["target"] = (c.shift(-5) > c).astype(int)

#     # Drop last 5 rows (no future label) and all NaN rows
#     df = df.iloc[:-5].dropna()

#     return df


# def feature_columns(df: pd.DataFrame) -> list:
#     """Return all feature column names (everything except OHLCV and target)."""
#     skip = {"Open", "High", "Low", "Close", "Volume", "target"}
#     return [c for c in df.columns if c not in skip]
"""
Feature engineering for the 5-day trend prediction model.

Covers:
  - Technical indicators  : RSI, MACD, Bollinger Bands, ATR, OBV, Stochastic, CCI,
                            Williams %R, Donchian Channel, VWAP proxy, ADX
  - Price/volume patterns : returns, gaps, volume ratio, price position, body ratio,
                            candle patterns, rolling skew, momentum divergence
  - Regime features       : trend strength, volatility regime, vol-adjusted momentum
  - Macro / sector proxy  : VIX (^VIX), Nifty 50 (^NSEI) relative strength
  - Sentiment proxy       : news-driven via yfinance .news headline count + polarity
                            (uses VADER if available, else binary title-count heuristic)
  - Interaction features  : RSI × trend, volume × momentum (non-linear combinations)
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


def _williams_r(high, low, close, period: int = 14) -> pd.Series:
    """Williams %R momentum oscillator"""
    high_max = high.rolling(period).max()
    low_min  = low.rolling(period).min()
    return -100 * (high_max - close) / (high_max - low_min).replace(0, np.nan)


def _adx(high, low, close, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend STRENGTH (not direction)"""
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr_raw   = _atr(high, low, close, period)
    plus_di   = 100 * pd.Series(plus_dm,  index=close.index).ewm(span=period, adjust=False).mean() / atr_raw.replace(0, np.nan)
    minus_di  = 100 * pd.Series(minus_dm, index=close.index).ewm(span=period, adjust=False).mean() / atr_raw.replace(0, np.nan)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


def _donchian_pct(high, low, close, period: int = 20) -> pd.Series:
    """Where is price within its N-day channel (0 = channel bottom, 1 = top)"""
    ch_high = high.rolling(period).max()
    ch_low  = low.rolling(period).min()
    return (close - ch_low) / (ch_high - ch_low).replace(0, np.nan)


def _vwap_proxy(high, low, close, volume) -> pd.Series:
    """Rolling 20-day VWAP proxy — price relative to volume-weighted average"""
    typical = (high + low + close) / 3
    vwap    = (typical * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    return (close - vwap) / vwap.replace(0, np.nan)


# ─────────────────────────────────────────────
# CANDLE PATTERN / BODY FEATURES
# ─────────────────────────────────────────────

def _candle_features(open_: pd.Series, high, low, close) -> pd.DataFrame:
    body   = (close - open_).abs()
    candle = (high - low).replace(0, np.nan)
    upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low
    return pd.DataFrame({
        "body_ratio":    body / candle,
        "upper_wick_r":  upper_wick / candle,
        "lower_wick_r":  lower_wick / candle,
        "candle_dir":    np.sign(close - open_),       # +1 green, -1 red
    }, index=close.index)


# ─────────────────────────────────────────────
# VOLATILITY REGIME
# ─────────────────────────────────────────────

def _vol_regime(returns: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    """Ratio of short-window vol to long-window vol — >1 means rising volatility"""
    short_vol = returns.rolling(short).std()
    long_vol  = returns.rolling(long).std()
    return short_vol / long_vol.replace(0, np.nan)


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
        # VIX regime: high fear = VIX > 20
        macro["vix_high"]  = (macro["vix_close"] > 20).astype(int)
    except Exception:
        macro["vix_close"] = np.nan
        macro["vix_chg"]   = np.nan
        macro["vix_high"]  = np.nan

    # NIFTY relative strength (skip if symbol IS nifty)
    if symbol not in ("^NSEI", "NIFTY50"):
        try:
            nifty = yf.Ticker("^NSEI").history(period=period)[["Close"]].rename(columns={"Close": "nifty"})
            nifty.index = nifty.index.tz_localize(None) if nifty.index.tz else nifty.index
            macro = macro.join(nifty, how="left")
            macro["nifty"] = macro["nifty"].ffill()
            macro["rel_strength"] = index_df["Close"].pct_change() - macro["nifty"].pct_change()
            # Rolling relative strength
            macro["rel_strength_5"] = macro["rel_strength"].rolling(5).sum()
            macro.drop(columns=["nifty"], inplace=True)
        except Exception:
            macro["rel_strength"]   = np.nan
            macro["rel_strength_5"] = np.nan
    else:
        macro["rel_strength"]   = 0.0
        macro["rel_strength_5"] = 0.0

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

    c, h, l, v, o = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]
    returns = c.pct_change()

    # ── Core Technical ──
    df["rsi_14"]        = _rsi(c, 14)
    df["rsi_7"]         = _rsi(c, 7)
    df["rsi_21"]        = _rsi(c, 21)
    macd, sig, hist     = _macd(c)
    df["macd"]          = macd
    df["macd_signal"]   = sig
    df["macd_hist"]     = hist
    df["macd_cross"]    = np.sign(hist) - np.sign(hist.shift(1))  # crossover signal
    _, _, bb_width, pct_b = _bollinger(c, 20)
    df["bb_width"]      = bb_width
    df["bb_pct_b"]      = pct_b
    df["bb_squeeze"]    = (bb_width < bb_width.rolling(20).quantile(0.2)).astype(int)
    df["atr_14"]        = _atr(h, l, c, 14)
    df["atr_pct"]       = df["atr_14"] / c.replace(0, np.nan)
    df["obv"]           = _obv(c, v)
    df["obv_chg"]       = df["obv"].pct_change(5)
    df["obv_signal"]    = np.sign(df["obv_chg"])
    pct_k, pct_d        = _stochastic(h, l, c)
    df["stoch_k"]       = pct_k
    df["stoch_d"]       = pct_d
    df["stoch_cross"]   = np.sign(pct_k - pct_d)
    df["cci_20"]        = _cci(h, l, c, 20)
    df["williams_r"]    = _williams_r(h, l, c, 14)
    df["adx"]           = _adx(h, l, c, 14)
    df["donchian_pct"]  = _donchian_pct(h, l, c, 20)
    df["vwap_dist"]     = _vwap_proxy(h, l, c, v)

    # ── Candle patterns ──
    candles = _candle_features(o, h, l, c)
    for col in candles.columns:
        df[col] = candles[col]

    # ── Price / Volume patterns ──
    df["ret_1"]         = returns
    df["ret_3"]         = c.pct_change(3)
    df["ret_5"]         = c.pct_change(5)
    df["ret_10"]        = c.pct_change(10)
    df["ret_20"]        = c.pct_change(20)
    df["gap"]           = (o - c.shift(1)) / c.shift(1).replace(0, np.nan)
    df["hl_range"]      = (h - l) / c.replace(0, np.nan)
    df["close_pos"]     = (c - l) / (h - l).replace(0, np.nan)
    df["vol_ratio_5"]   = v / v.rolling(5).mean().replace(0, np.nan)
    df["vol_ratio_20"]  = v / v.rolling(20).mean().replace(0, np.nan)
    df["vol_spike"]     = (df["vol_ratio_5"] > 2.0).astype(int)
    df["price_vs_sma20"]= c / c.rolling(20).mean().replace(0, np.nan) - 1
    df["price_vs_sma50"]= c / c.rolling(50).mean().replace(0, np.nan) - 1
    df["price_vs_ema20"]= c / c.ewm(span=20, adjust=False).mean().replace(0, np.nan) - 1
    df["rolling_std_10"]= returns.rolling(10).std()
    df["rolling_skew"]  = returns.rolling(20).skew()    # asymmetry signal
    df["vol_regime"]    = _vol_regime(returns)

    # ── Momentum consistency features ──
    df["consec_up"]     = returns.gt(0).rolling(5).sum()   # consecutive up days in window
    df["consec_down"]   = returns.lt(0).rolling(5).sum()
    df["mom_5_20"]      = df["ret_5"] - df["ret_20"]       # short vs long momentum

    # ── Interaction features (non-linear combinations improve tree models) ──
    df["rsi_vol"]       = df["rsi_14"] * df["vol_ratio_5"]  # RSI weighted by volume
    df["macd_rsi"]      = df["macd_hist"] * (df["rsi_14"] / 50)
    df["bb_rsi"]        = df["bb_pct_b"] * (df["rsi_14"] / 100)
    df["vol_adx"]       = df["vol_ratio_5"] * df["adx"] / 100

    # ── Macro ──
    macro = _macro_features(symbol, period, df)
    df = df.join(macro, how="left")

    # ── Sentiment (single scalar added as constant column, refreshed at predict time) ──
    df["sentiment"] = _sentiment_score(ticker)
    df["sent_vol"]  = df["sentiment"] * df["vol_ratio_5"]  # sentiment × volume interaction

    # ── Target: 1 if price is higher 5 days later ──
    df["target"] = (c.shift(-5) > c).astype(int)

    # Drop last 5 rows (no future label) and all NaN rows
    df = df.iloc[:-5].dropna()

    return df


def feature_columns(df: pd.DataFrame) -> list:
    """Return all feature column names (everything except OHLCV and target)."""
    skip = {"Open", "High", "Low", "Close", "Volume", "target"}
    return [c for c in df.columns if c not in skip]
