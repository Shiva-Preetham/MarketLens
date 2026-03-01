from pydantic import BaseModel


class StockCreate(BaseModel):
    symbol: str
    company_name: str
    market: str
    sector: str


class StockResponse(BaseModel):
    id: int
    symbol: str
    company_name: str
    market: str
    sector: str

    class Config:
        from_attributes = True