from fastapi import APIRouter
from app.quant.portfolio import (
    optimize_portfolio,
    minimum_variance_portfolio,
)
from app.services.local_llm_service import explain_portfolio_local

router = APIRouter()


# =========================
# MAX SHARPE PORTFOLIO
# =========================
@router.post("/portfolio/optimize")
def optimize(tickers: list[str]):
    return optimize_portfolio(tickers)


# =========================
# MINIMUM VARIANCE PORTFOLIO
# =========================
@router.post("/portfolio/min_variance")
def min_variance(tickers: list[str]):
    return minimum_variance_portfolio(tickers)


# =========================
# OPTIMIZE + AI EXPLANATION (MAIN PRODUCT ENDPOINT)
# =========================
@router.post("/portfolio/optimize_ai")
def optimize_ai(tickers: list[str]):
    metrics = optimize_portfolio(tickers)
    explanation = explain_portfolio_local(metrics)

    return {
        "metrics": metrics,
        "explanation": explanation
    }


# =========================
# STANDALONE AI EXPLANATION (OPTIONAL)
# =========================
@router.post("/portfolio/explain_local")
def explain_local(metrics: dict):
    explanation = explain_portfolio_local(metrics)
    return {"explanation": explanation}