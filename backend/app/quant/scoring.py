def normalize_trend(trend):
    return 100 if trend == "Bullish" else 40


def normalize_rsi(rsi):
    score = 100 - abs(rsi - 50) * 2
    return max(0, min(score, 100))


def normalize_sharpe(sharpe):
    score = sharpe * 40
    return max(0, min(score, 100))


def normalize_drawdown(drawdown):
    score = 100 - abs(drawdown) * 500
    return max(0, min(score, 100))


def calculate_composite_score(trend, rsi, sharpe, drawdown):
    trend_score = normalize_trend(trend)
    rsi_score = normalize_rsi(rsi)
    sharpe_score = normalize_sharpe(sharpe)
    drawdown_score = normalize_drawdown(drawdown)

    final_score = (
        trend_score * 0.3 +
        sharpe_score * 0.3 +
        rsi_score * 0.2 +
        drawdown_score * 0.2
    )

    return round(final_score, 2)


def generate_rating(score):
    if score >= 75:
        return "Strong Buy"
    elif score >= 60:
        return "Buy"
    elif score >= 45:
        return "Hold"
    elif score >= 30:
        return "Weak"
    else:
        return "Avoid"