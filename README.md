# Project Blueprint

## Product Goal

MarketLens is a market intelligence platform that combines stock analysis, news sentiment, and ML prediction in one dashboard.

## Existing Strengths

- FastAPI backend
- stock and risk metric endpoints
- watchlist storage
- portfolio utilities
- feature engineering for market prediction
- Random Forest + XGBoost prediction pipeline

## Gaps Being Fixed

- sentiment analysis needs to be visible in the main frontend
- prediction endpoints need to be easier to access from the UI
- news sentiment should become a first-class data feature
- future dashboard exports should be structured for Power BI

## Current Architecture

### Frontend
- `frontend/index.html`
  - main UI for stock analysis, portfolio, watchlist, sentiment analysis, and ML prediction

### Backend
- `backend/app/main.py`
  - FastAPI app and router registration
- `backend/app/routes/intelligence.py`
  - sentiment endpoint
- `backend/app/services/market_intelligence_service.py`
  - headline scoring and sentiment aggregation
- `backend/app/quant/ml_model.py`
  - tree-based prediction model
- `backend/app/quant/features.py`
  - technical, macro, and sentiment features

## Phase Roadmap

### Phase 1: Main UI Integration
- add sentiment analysis under the existing ML Prediction tab
- keep existing dashboard, portfolio, and watchlist behavior unchanged
- remove separate demo frontend

### Phase 2: Persistent Data Layer
- store news articles
- store sentiment snapshots
- store prediction runs
- create analytical tables for dashboarding

### Phase 3: Better Sentiment Model
- replace lightweight scoring with FinBERT
- compare sentiment score changes over time
- connect sentiment aggregates into the ML feature set more explicitly

### Phase 4: Dashboard And Deployment
- add Docker
- create dashboard-ready exports
- build Power BI reports
- add basic monitoring/logging

## Tool Choices

- FastAPI for APIs
- SQLAlchemy for database access
- yfinance for market data and available news metadata
- scikit-learn and XGBoost for ML prediction
- lightweight sentiment scoring now, FinBERT later
- Power BI later for business-facing dashboards

Built by Shiva Preetham

This project is for educational and portfolio purposes.
