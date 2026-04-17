# MarketLens Intelligence

MarketLens is a **Market Intelligence + News Sentiment + ML Prediction** platform for stock analysis.

It combines:
- live market data
- risk and technical analysis
- news sentiment analysis
- 5-day market direction prediction
- dashboard-ready API responses

## Product Flow

1. Fetch market data for a ticker.
2. Calculate risk and technical metrics.
3. Fetch recent news headlines.
4. Convert headlines into sentiment signals.
5. Run the ML prediction pipeline.
6. Show results in the existing frontend dashboard.

## Current Features

- FastAPI backend
- SQLAlchemy database layer
- SQLite fallback for simple local runs
- stock watchlist APIs
- market data APIs using `yfinance`
- risk metrics and quant analysis
- news sentiment endpoint at `/intelligence/{ticker}`
- ML prediction endpoints at `/predict/{ticker}`
- integrated ML Prediction + Sentiment section inside `frontend/index.html`

## Project Architecture

### Backend
- `backend/app/main.py`
  - FastAPI app setup and router registration
- `backend/app/routes/`
  - API endpoints for stocks, analysis, portfolio, prediction, and intelligence
- `backend/app/services/`
  - service logic for market data, sentiment, ML, and portfolio workflows
- `backend/app/quant/`
  - indicators, features, risk logic, portfolio math, and ML model code

### Frontend
- `frontend/index.html`
  - main dashboard UI with stock analysis, portfolio, watchlist, ML prediction, and news sentiment

### Docs
- `docs/project_blueprint.md`
  - architecture and phase roadmap
- `docs/application_positioning.md`
  - project positioning, resume bullets, and LinkedIn description

## Local Run

### Backend
```bash
cd backend
uvicorn app.main:app --reload
```

### Frontend
Open:

```text
frontend/index.html
```

Then go to the **ML Predict** tab.

## Key Endpoints

- `/`
- `/health`
- `/analyze/{ticker}`
- `/stocks/market-data?symbol=...`
- `/stocks/risk-metrics?symbol=...`
- `/intelligence/{ticker}`
- `/predict/{ticker}`
- `/predict/{ticker}/train`
- `/predict/{ticker}/info`

## Next Planned Improvements

- persist sentiment snapshots in the database
- export dashboard-ready datasets
- upgrade sentiment scoring to FinBERT
- add Docker setup
- add Power BI dashboard outputs
