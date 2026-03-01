from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app import models
from app.schemas import StockCreate, StockResponse

router = APIRouter()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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


@router.get("/stocks/", response_model=list[StockResponse])
def get_stocks(db: Session = Depends(get_db)):
    return db.query(models.Stock).all()