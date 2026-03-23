"""China A-share local CSV loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_CN_RENAME_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover_rate",
}


def _normalize_stock_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.rename(columns=_CN_RENAME_MAP).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Stock CSV missing required columns: {missing}")

    d["date"] = pd.to_datetime(d["date"], errors="coerce", dayfirst=False)
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    return d.dropna(subset=["close"]).reset_index(drop=True)


def load_stock_from_csv(code: str, csv_dir: str | Path) -> pd.DataFrame:
    """Load a single stock from local CSV and normalize to OHLCV schema."""
    path = Path(csv_dir) / f"{code}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stock CSV not found: {path}")
    raw_df = pd.read_csv(path)
    out = _normalize_stock_df(raw_df)
    out["symbol"] = code
    return out


def load_stock_list(list_file: str | Path) -> list[str]:
    path = Path(list_file)
    if not path.exists():
        raise FileNotFoundError(f"Stock list file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
