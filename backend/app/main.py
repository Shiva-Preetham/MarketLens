from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine
from app import models

from app.routes.stocks import router as stocks_router
from app.routes.analyze import router as analyze_router
from app.routes.portfolio import router as portfolio_router

app = FastAPI()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables
models.Base.metadata.create_all(bind=engine)

# Root endpoint
@app.get("/")
def root():
    return {"message": "MarketLens Backend Running 🚀"}

# Register routers
app.include_router(stocks_router)
app.include_router(analyze_router)
app.include_router(portfolio_router)