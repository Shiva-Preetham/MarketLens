# Project Blueprint

## Repo Audit

### What already existed
- FastAPI backend with routes for stock CRUD, analysis, portfolio math, and ML prediction
- feature engineering and a tree-based market-direction model in `backend/app/quant/`
- a portfolio-oriented frontend prototype in `frontend/index.html`

### Gaps found during audit
- the ML prediction router existed but was not registered in `main.py`
- there was no true market-intelligence/news-sentiment module yet
- backend startup depended on `DATABASE_URL`, which made local onboarding fragile
- the UI story was broader than the backend story and had some prototype-level rough edges

### What we should keep
- quant feature engineering
- stock/risk analysis endpoints
- portfolio optimization code as supporting material
- the existing ML model as a useful phase-1 baseline

### What we should redesign over time
- data persistence beyond a simple watchlist table
- the sentiment pipeline so it becomes a first-class dataset, not just a utility
- the frontend into a cleaner product surface
- model evaluation and monitoring flow

## Target Project Story

Build MarketLens into an internship-ready:

**Market Intelligence + News Sentiment + ML Prediction + Dashboard project**

Core value:
- turn raw market and news signals into clean, explainable insights
- expose them through APIs
- make them dashboard-ready
- show both analytics and ML engineering skills in one repo

## Recommended Architecture

### Presentation layer
- FastAPI docs for API exploration
- lightweight demo frontend for manual testing
- later: Power BI dashboard on exported/model-ready data

### Service layer
- stock analysis service
- market intelligence service
- ML prediction service
- later: dataset export service

### Data layer
- watchlist storage in SQL
- local analytical storage for snapshots and derived tables in later phases
- eventual bronze/silver/gold style modeling for market/news data

### ML layer
- baseline and production model comparison
- current model: RandomForest + XGBoost ensemble
- recommended long-term production model: XGBoost primary, simpler baselines retained for comparison

## Final Stack Recommendation

### Must-have now
- Python
- FastAPI
- SQLAlchemy
- SQLite or Postgres
- pandas
- yfinance
- scikit-learn
- xgboost
- VADER or heuristic sentiment for the free phase
- Power BI later for visualization

### Strong next upgrades
- DuckDB for analytical extracts
- MLflow for experiments
- Docker for reproducible setup
- Prometheus/Grafana or lightweight observability
- FinBERT for better financial sentiment

### Skip for now
- Hadoop
- paid observability tools
- Kubernetes in the first implementation phase

## Phase Plan

## Phase 1: Foundation + First Real Data Product

### What we built
- default local database fallback
- wired the prediction router into the main app
- added health check
- added market-intelligence/news-sentiment endpoint
- added a simpler demo page for the new story

### Why it matters
- the project now has a clearer product identity
- you can demo both data ingestion and ML-backed reasoning
- the backend is easier to run locally

### Interview framing
- “I first fixed the platform basics so the project was reliable to run.”
- “Then I added a market-intelligence module because it connects unstructured news to structured analysis.”

## Phase 2: Data Product Layer

### Goals
- persist news snapshots
- persist sentiment aggregates
- create exportable datasets for BI tools
- make the project feel more like a real analytics platform

### Suggested deliverables
- `news_articles`
- `sentiment_snapshots`
- `prediction_runs`
- `market_daily_features`

## Phase 3: Better Modeling

### Goals
- compare logistic regression, random forest, and xgboost
- add model evaluation report endpoints
- upgrade to FinBERT for finance-specific sentiment
- explicitly combine sentiment + technical features

### Interview framing
- “I kept a simple baseline so I could measure whether the more advanced model actually helped.”

## Phase 4: Internship Polish

### Goals
- Dockerize the app
- add CI checks
- add a Power BI dashboard
- improve README screenshots and demo flow

## Module-by-Module Teaching Notes

## `backend/app/database.py`

### What it does
- creates the SQLAlchemy engine and session

### Why we chose it
- it keeps the project standard and easy to explain
- SQLite fallback lowers setup friction for recruiters and interview demos

### Likely interviewer question
- “Why use SQLite fallback?”

### Strong answer
- “For local development and demos, SQLite removes setup friction. The abstraction still supports switching to Postgres later through the same SQLAlchemy layer.”

## `backend/app/services/market_intelligence_service.py`

### What it does
- fetches recent ticker news metadata
- scores headlines
- aggregates headline sentiment into a single signal

### Why we chose it
- it creates a bridge between raw headlines and a dashboard/ML-friendly feature
- it is free, lightweight, and easy to iterate on

### Likely interviewer question
- “Why not use a large transformer immediately?”

### Strong answer
- “I started with a lightweight free sentiment layer to validate the product flow end to end. Once the pipeline and interfaces are stable, the next upgrade is FinBERT for stronger finance-domain sentiment.”

## `backend/app/routes/intelligence.py`

### What it does
- exposes the sentiment pipeline through a clean API

### Why we chose it
- APIs make the intelligence module reusable by dashboards, frontends, and later automated jobs

### Likely interviewer question
- “Why separate routes and services?”

### Strong answer
- “I wanted thin routes and service-level business logic so the code stays easier to test, extend, and reason about.”

## `frontend/internship_demo.html`

### What it does
- gives a simple, cleaner interface for phase-1 demos

### Why we chose it
- the original UI is still prototype-heavy, so a focused demo page is a safer way to show the new product story without breaking everything at once

### Likely interviewer question
- “Why add a separate demo page?”

### Strong answer
- “I wanted a stable demonstration surface for the new intelligence workflow while preserving the original prototype during the transition.”
