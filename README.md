# MarketLens

MarketLens 🚀
Backend Architecture

MarketLens is a stock market analysis backend built using:

FastAPI

PostgreSQL

SQLAlchemy ORM

Pydantic Validation

Swagger Auto Docs

Git Version Control

Current Features

Create stock records

Retrieve stock records

PostgreSQL persistent storage

REST API structure

Clean project architecture

Tech Stack

Python 3.12+

FastAPI

PostgreSQL

SQLAlchemy

Uvicorn

Run Locally
git clone https://github.com/Shiva-Preetham/MarketLens.git
cd MarketLens
pip install -r requirements.txt
cd backend
uvicorn app.main:app --reload

Visit:

http://127.0.0.1:8000/docs

Then:

git add README.md
git commit -m "Updated README with backend documentation"
git push
✅ Step 3 — Add .env Support (Optional But Smart)

Later we’ll move your DB URL into .env.

But for now your repo is clean.

✅ Step 4 — Create a Version Tag (Optional Advanced Move)

Run:

git tag v0.1
git push origin v0.1

Research labs

System Architecture

Quant Methodology

AI Layer

API Endpoints
## Architecture

Frontend (HTML + Chart.js)
        ↓
FastAPI Backend
        ↓
Data Service (yfinance)
        ↓
Analysis Engine (RSI, Sharpe, Risk)
        ↓
Optional LLM Commentary


more Machine Learning Features,
