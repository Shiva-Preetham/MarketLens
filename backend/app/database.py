from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ⚠️ Replace YOUR_PASSWORD with your actual postgres password
DATABASE_URL = "postgresql://postgres:spch27@localhost:5432/marketlens"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()