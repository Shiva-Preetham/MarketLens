# MarketLens Intelligence

MarketLens is being upgraded from a stock-analysis prototype into an internship-ready **Market Intelligence + News Sentiment + ML Prediction** project.

The project is designed to help demonstrate:
- Python backend engineering
- SQL and data modeling fundamentals
- API design with FastAPI
- market data processing
- news sentiment analysis
- interpretable ML prediction
- dashboard-ready outputs for BI tools like Power BI

## Why This Project Matters

This repo is aimed at internship applications for analytics, data, and ML-focused roles. The product story is simple:

1. pull market data
2. pull recent news metadata
3. turn headlines into a sentiment signal
4. combine quantitative features with ML prediction
5. expose everything through clean APIs that a dashboard can consume

## Current Phase

Phase 1 has been implemented in this repo:
- improved backend startup reliability with a default local SQLite database fallback
- wired the existing ML prediction router into the FastAPI app
- added `/health` for quick API verification
- added `/intelligence/{ticker}` for market-intelligence and news-sentiment summaries
- added a clean demo page at [frontend/internship_demo.html](C:/Users/shiva/Desktop/MarketLens/frontend/internship_demo.html)

## Project Architecture

### Backend
- `backend/app/main.py`
  - app entrypoint and router registration
- `backend/app/routes/`
  - API endpoints for stocks, analysis, portfolio, prediction, and intelligence
- `backend/app/services/`
  - business logic for market data, news sentiment, ML, and portfolio workflows
- `backend/app/quant/`
  - feature engineering, indicators, portfolio math, and ML model code

### Frontend
- `frontend/index.html`
  - legacy UI prototype
- `frontend/internship_demo.html`
  - cleaner phase-1 demo page for intelligence and prediction

### Docs
- [docs/project_blueprint.md](C:/Users/shiva/Desktop/MarketLens/docs/project_blueprint.md)
  - audit, architecture, roadmap, interview framing
- [docs/application_positioning.md](C:/Users/shiva/Desktop/MarketLens/docs/application_positioning.md)
  - project title, resume bullets, README plan, LinkedIn summary, talking points

## Phase Roadmap

### Phase 1: Foundation + Market Intelligence
- reliable backend startup
- health endpoint
- news headline sentiment endpoint
- visible demo page

### Phase 2: Better Data Product
- store market and news snapshots in a warehouse-style schema
- add richer feature tables
- create dashboard-ready endpoints and extracts
- improve frontend experience

### Phase 3: Stronger ML
- evaluate baseline vs tree-based models
- add better validation and diagnostics
- upgrade sentiment to FinBERT
- connect sentiment features into the prediction pipeline more explicitly

### Phase 4: Deployment + Resume Polish
- Docker setup
- CI checks
- observability basics
- Power BI dashboard
- final README and demo assets

## Local Run

### Backend
```bash
cd backend
uvicorn app.main:app --reload
```

### Demo Frontend
Open [frontend/internship_demo.html](C:/Users/shiva/Desktop/MarketLens/frontend/internship_demo.html) in a browser after the backend is running.

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

## How To Explain It In Interviews

Short version:

> I built a market intelligence platform that combines price-based analysis, headline sentiment, and an ML prediction API. I focused on turning raw finance data into clean backend services and dashboard-ready outputs rather than only training a model in a notebook.

For a deeper breakdown, use:
- [docs/project_blueprint.md](C:/Users/shiva/Desktop/MarketLens/docs/project_blueprint.md)
- [docs/application_positioning.md](C:/Users/shiva/Desktop/MarketLens/docs/application_positioning.md)
