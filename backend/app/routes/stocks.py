from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app import models
from app.schemas import StockCreate, StockResponse
from app.services.data_service import fetch_stock_data
from app.services.data_service import calculate_risk_metrics

router = APIRouter()


# ---------------------------
# Database Dependency
# ---------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------
# Create Stock (DB)
# ---------------------------
@router.post("/stocks/", response_model=StockResponse)
def create_stock(stock: StockCreate, db: Session = Depends(get_db)):
    new_stock = models.Stock(
        symbol=stock.symbol,
        company_name=stock.company_name,
        market=stock.market,
        sector=stock.sector
    )
    db.add(new_stock)
    db.commit()
    db.refresh(new_stock)
    return new_stock


# ---------------------------
# Get All Stored Stocks (DB)
# ---------------------------
@router.get("/stocks/", response_model=list[StockResponse])
def get_stocks(db: Session = Depends(get_db)):
    return db.query(models.Stock).all()


# ---------------------------
# Fetch Live Market Data (Yahoo Finance)
# ---------------------------
@router.get("/stocks/market-data")
def get_market_data(symbol: str, period: str = "1y"):
    return fetch_stock_data(symbol, period)
@router.get("/stocks/risk-metrics")

def get_risk_metrics(symbol: str, period: str = "1y"):
    return calculate_risk_metrics(symbol, period)