from sqlalchemy import Column, Integer, String
from .database import Base

class Stock(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    company_name = Column(String)
    market = Column(String)
    sector = Column(String)