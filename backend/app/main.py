from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from .database import engine, SessionLocal
from . import models
from pydantic import BaseModel

app = FastAPI()

models.Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {"message": "MarketLens Backend Running 🚀"}


# Add new stock


class StockCreate(BaseModel):
    symbol: str
    company_name: str
    market: str
    sector: str


@app.post("/stocks/")
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


# Get all stocks
@app.get("/stocks/")
def get_stocks(db: Session = Depends(get_db)):
    return db.query(models.Stock).all()