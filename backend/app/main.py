from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from .database import engine, SessionLocal
from . import models
from pydantic import BaseModel

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class StockResponse(BaseModel):
    id: int
    symbol: str
    company_name: str
    market: str
    sector: str

    class Config:
        from_attributes = True


@app.post("/stocks/", response_model=StockResponse)
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
@app.get("/stocks/", response_model=list[StockResponse])
def get_stocks(db: Session = Depends(get_db)):
    return db.query(models.Stock).all()