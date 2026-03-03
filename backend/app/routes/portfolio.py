from fastapi import APIRouter, UploadFile, File
from app.quant.portfolio import (
    optimize_portfolio,
    minimum_variance_portfolio,
)
from app.quant.efficient_frontier import generate_efficient_frontier
from app.services.local_llm_service import explain_portfolio_local
from app.services.portfolio_import_service import parse_portfolio_file

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
# EFFICIENT FRONTIER
# =========================
@router.post("/portfolio/efficient_frontier")
def efficient_frontier(tickers: list[str], num_portfolios: int = 250):
    return generate_efficient_frontier(tickers, num_portfolios=num_portfolios)


# =========================
# IMPORT PORTFOLIO FILE(S)
# =========================
@router.post("/portfolio/import")
async def import_portfolio(file: list[UploadFile] = File(...)):
    """
    Accept one or more files and return combined holdings.
    FastAPI will send multiple parts with the same field name `file`.
    """
    all_holdings = []
    sources: set[str] = set()

    for f in file:
        data = await f.read()
        parsed = parse_portfolio_file(f.filename or "upload", data)
        sources.add(parsed.get("source_type", "unknown"))
        all_holdings.extend(parsed.get("holdings", []))

    return {
        "source_type": list(sources),
        "files_count": len(file),
        "holdings": all_holdings,
    }


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