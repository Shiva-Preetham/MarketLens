from fastapi import APIRouter
from app.services.analysis_service import analyze_stock

router = APIRouter()

@router.get("/analyze/{ticker}")
def analyze(ticker: str):
    return analyze_stock(ticker)