import io
from typing import List, Dict, Tuple

import pandas as pd


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


def parse_image_bytes(data: bytes) -> Tuple[List[Dict], str]:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        # OCR not configured on the server
        return [], "image_unsupported"

    image = Image.open(io.BytesIO(data))
    text = pytesseract.image_to_string(image)

    rows: List[Dict] = []
    for line in text.splitlines():
        parts = [p for p in line.replace("\t", " ").split(" ") if p]
        if not parts:
            continue
        symbol = parts[0].upper()
        # Very simple heuristic: SYMBOL QTY PRICE
        qty = None
        avg = None
        if len(parts) >= 2:
            try:
                qty = float(parts[1])
            except ValueError:
                qty = None
        if len(parts) >= 3:
            try:
                avg = float(parts[2])
            except ValueError:
                avg = None
        rows.append({"symbol": symbol, "qty": qty, "avg": avg})

    return rows, "image_ocr"


def parse_portfolio_file(filename: str, data: bytes) -> Dict:
    ext = ""
    if "." in filename:
        ext = filename[filename.rfind(".") :].lower()

    holdings, source = parse_tabular_bytes(data, ext)

    if not holdings and ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        holdings, source = parse_image_bytes(data)

    return {
        "source_type": source,
        "holdings": holdings,
    }

