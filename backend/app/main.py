import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import models
from app.database import engine
from app.routes.analyze import router as analyze_router
from app.routes.intelligence import router as intelligence_router
from app.routes.portfolio import router as portfolio_router
from app.routes.predict import router as predict_router
from app.routes.stocks import router as stocks_router

app = FastAPI(
    title="MarketLens API",
    version="0.2.0",
    description=(
        "Market intelligence backend with quant analysis, "
        "news sentiment, watchlists, and ML prediction endpoints."
    ),
)

frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")
allow_origins = ["*"] if frontend_origin.strip() == "*" else [
    origin.strip() for origin in frontend_origin.split(",") if origin.strip()
]

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables
models.Base.metadata.create_all(bind=engine)


@app.get("/")
def root():
    return {
        "message": "MarketLens API is running",
        "product": "MarketLens Intelligence",
        "version": app.version,
        "docs": "/docs",
        "capabilities": [
            "stock analysis",
            "market data",
            "watchlist storage",
            "news sentiment intelligence",
            "5-day ML prediction",
        ],
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "database": "connected", "version": app.version}


# Register routers
app.include_router(stocks_router)
app.include_router(analyze_router)
app.include_router(intelligence_router)
app.include_router(portfolio_router)
app.include_router(predict_router)
