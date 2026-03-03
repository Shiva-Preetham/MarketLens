import io
import base64
import json
import os
import re
from typing import List, Dict, Tuple

import pandas as pd
import requests


# =========================
# TABULAR COLUMN NORMALIZER
# =========================
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_cols = {c.lower().strip(): c for c in df.columns}

    symbol_col = None
    for key in ("symbol", "ticker", "stock", "code"):
        if key in lower_cols:
            symbol_col = lower_cols[key]
            break

    qty_col = None
    for key in ("qty", "quantity", "shares", "units"):
        if key in lower_cols:
            qty_col = lower_cols[key]
            break

    price_col = None
    for key in ("avg_price", "average_price", "price", "buy_price", "cost"):
        if key in lower_cols:
            price_col = lower_cols[key]
            break

    cols = {}
    if symbol_col:
        cols["symbol"] = df[symbol_col].astype(str).str.strip()
    if qty_col:
        cols["qty"] = pd.to_numeric(df[qty_col], errors="coerce")
    if price_col:
        cols["avg"] = pd.to_numeric(df[price_col], errors="coerce")

    return pd.DataFrame(cols)


def _rows_from_frame(df: pd.DataFrame) -> List[Dict]:
    rows = []
    for _, row in df.iterrows():
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        qty = row.get("qty")
        avg = row.get("avg")
        rows.append(
            {
                "symbol": symbol,
                "qty": float(qty) if pd.notna(qty) else None,
                "avg": float(avg) if pd.notna(avg) else None,
            }
        )
    return rows


# =========================
# TABULAR FILE PARSER
# =========================
def parse_tabular_bytes(data: bytes, ext: str) -> Tuple[List[Dict], str]:
    ext = ext.lower()
    if ext in (".csv", ".txt"):
        df = pd.read_csv(io.StringIO(data.decode("utf-8", errors="ignore")))
        df_norm = _normalize_columns(df)
        return _rows_from_frame(df_norm), "csv"

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(data))
        df_norm = _normalize_columns(df)
        return _rows_from_frame(df_norm), "excel"

    if ext in (".json",):
        obj = pd.read_json(io.BytesIO(data))
        df_norm = _normalize_columns(obj)
        return _rows_from_frame(df_norm), "json"

    return [], "unknown"


# =========================
# CLAUDE VISION IMAGE PARSER
# Replaces broken pytesseract OCR
# =========================
def parse_image_bytes_with_claude(data: bytes, media_type: str = "image/jpeg") -> Tuple[List[Dict], str]:
    """
    Use Claude Vision (claude-sonnet-4-20250514) to extract holdings from
    a broker screenshot. Returns (holdings_list, source_type).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return [], "image_error_no_api_key"

    image_b64 = base64.standard_b64encode(data).decode("utf-8")

    prompt = """This is a screenshot from a stock broker app showing a holdings/portfolio list.

Extract every stock holding you can see. For each holding return:
- symbol: the stock ticker/symbol (e.g. GOLDBEES, BANKBETA)
- qty: the quantity / number of shares (look for "Qty." label)
- avg: the average buy price (look for "Avg." label)

Return ONLY a valid JSON array, nothing else. Example:
[{"symbol": "GOLDBEES", "qty": 2, "avg": 71.81}, {"symbol": "BANKBETA", "qty": 2, "avg": 50.13}]

If a value is not visible, use null. Do not include any explanation or markdown."""

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        text = response.json()["content"][0]["text"].strip()

        # Strip markdown fences if present
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

        raw = json.loads(text)
        holdings: List[Dict] = []
        for item in raw:
            symbol = str(item.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            qty = item.get("qty")
            avg = item.get("avg")
            holdings.append({
                "symbol": symbol,
                "qty": float(qty) if qty is not None else None,
                "avg": float(avg) if avg is not None else None,
            })
        return holdings, "image_claude_vision"

    except Exception as e:
        return [], f"image_error:{str(e)}"


# =========================
# MEDIA TYPE HELPER
# =========================
_EXT_TO_MEDIA_TYPE = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".gif":  "image/gif",
    ".bmp":  "image/png",   # convert-free fallback
}

IMAGE_EXTENSIONS = set(_EXT_TO_MEDIA_TYPE.keys())


# =========================
# MAIN ENTRY POINT
# =========================
def parse_portfolio_file(filename: str, data: bytes) -> Dict:
    ext = ""
    if "." in filename:
        ext = filename[filename.rfind("."):].lower()

    # Try tabular formats first
    holdings, source = parse_tabular_bytes(data, ext)
    if holdings:
        return {"source_type": source, "holdings": holdings}

    # Fall back to Claude Vision for images
    if ext in IMAGE_EXTENSIONS:
        media_type = _EXT_TO_MEDIA_TYPE.get(ext, "image/jpeg")
        holdings, source = parse_image_bytes_with_claude(data, media_type)
        return {"source_type": source, "holdings": holdings}

    return {"source_type": "unsupported", "holdings": []}