# Project Positioning

## Project Title

**MarketLens Intelligence: News Sentiment and ML-Powered Market Analysis Platform**

## One-Line Summary

MarketLens turns live market data and recent news headlines into sentiment signals, risk metrics, and short-term ML predictions through a FastAPI backend and web dashboard.

## Resume Bullets

- Built a FastAPI-based market intelligence platform that combines stock analysis, news sentiment scoring, and ML prediction APIs using Python, pandas, SQLAlchemy, scikit-learn, and XGBoost.
- Added a news sentiment pipeline that fetches recent ticker headlines, scores sentiment, aggregates positive/neutral/negative signals, and displays them inside the main dashboard.
- Integrated a 5-day market direction prediction workflow with model diagnostics, probability outputs, feature snapshots, and retraining endpoints.

## LinkedIn Description

Built MarketLens Intelligence, a market analysis project that combines live stock data, risk metrics, news sentiment, and ML-based short-term prediction in one dashboard. The backend uses FastAPI and SQLAlchemy, while the ML layer uses engineered financial features with tree-based models. The project is designed as a practical data product with APIs that can later feed Power BI dashboards and stronger NLP models such as FinBERT.

## Skills Covered

- Python
- FastAPI
- SQLAlchemy
- SQLite/Postgres-ready database layer
- pandas
- yfinance
- scikit-learn
- XGBoost
- sentiment analysis
- feature engineering
- REST API design
- frontend integration
- dashboard-ready data design

## Short Project Explanation

MarketLens Intelligence pulls stock data, calculates quant metrics, fetches recent news headlines, converts those headlines into sentiment signals, and runs a short-term ML prediction model. The main frontend now shows sentiment analysis and prediction results under the ML Prediction section, so the product has one clear flow from data to signal.

## Current Limitations

- sentiment scoring is lightweight in the current phase
- FinBERT is planned for a later upgrade
- sentiment snapshots are not persisted yet
- Power BI exports are planned but not implemented yet
