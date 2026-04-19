import os

import requests


def explain_portfolio_local(metrics: dict):
    ollama_url = os.getenv("OLLAMA_URL")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:4b")

    # Hosted deployments generally do not have a local Ollama daemon.
    if os.getenv("PORT") and not ollama_url:
        return "AI explanation unavailable: OLLAMA_URL is not configured for this deployment."

    ollama_url = ollama_url or "http://localhost:11434/api/generate"

    prompt = f"""
You are a senior quantitative portfolio strategist.

Provide a concise institutional-grade analysis of this portfolio.

Rules:
- Maximum 5–7 sentences
- No emojis
- No fluff
- No markdown formatting
- Be direct and analytical
- Focus only on investment relevance

Metrics:
{metrics}

Structure your response as:
1. Return outlook
2. Risk assessment
3. Sharpe interpretation
4. Investor suitability
5. One clear recommendation
"""

    try:
        response = requests.post(
            ollama_url,
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3   # more analytical, less creative
            },
            timeout=20
        )

        response.raise_for_status()
        return response.json()["response"]

    except Exception as e:
        return f"AI explanation unavailable: {str(e)}"
