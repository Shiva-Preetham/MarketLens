from fastapi import APIRouter, Query

from app.services.market_intelligence_service import build_market_intelligence

router = APIRouter(prefix="/intelligence", tags=["intelligence"])


@router.get("/{ticker}")
def get_market_intelligence(
    ticker: str,
    limit: int = Query(default=8, ge=1, le=20),
):
    """
    Return a lightweight market-intelligence snapshot for a ticker.

    Phase 1 is intentionally practical:
    - fetch recent ticker news metadata
    - score headlines with free sentiment logic
    - return a UI-friendly summary for dashboards and product screens
    """
    return build_market_intelligence(ticker, limit=limit)
