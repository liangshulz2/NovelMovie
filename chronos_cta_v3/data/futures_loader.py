"""Futures market data loader."""

from __future__ import annotations

import time
from pathlib import Path

import akshare as ak
import pandas as pd

_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "futures"


def _cache_path(symbol: str, start_date: str | None, end_date: str | None) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    start = start_date or "all"
    end = end_date or "all"
    return _CACHE_DIR / f"{symbol}_{start}_{end}.parquet"


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume", "hold"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cleaned = df.dropna(subset=["close"])
    if cleaned.empty:
        raise ValueError("Loaded futures data is empty after numeric cleaning and close-price filtering.")
    return cleaned


def load_futures(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    retry: int = 3,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load Chinese futures daily bars from Sina via AkShare with retries and local cache."""
    cache_file = _cache_path(symbol, start_date, end_date)
    if use_cache and cache_file.exists():
        cached = pd.read_parquet(cache_file)
        if cached.empty:
            raise ValueError(f"Cached futures data for {symbol} is empty: {cache_file}")
        return _clean_df(cached)

    last_err: Exception | None = None
    for attempt in range(1, retry + 1):
        try:
            df = ak.futures_zh_daily_sina(symbol=symbol)
            if start_date is not None:
                df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(start_date)]
            if end_date is not None:
                df = df[pd.to_datetime(df["date"]) <= pd.Timestamp(end_date)]

            cleaned = _clean_df(df)
            if use_cache:
                cleaned.to_parquet(cache_file, index=False)
            return cleaned
        except Exception as exc:  # noqa: PERF203
            last_err = exc
            if attempt < retry:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(
        f"Failed to load futures data for {symbol} after {retry} attempts. "
        f"Last error: {last_err}"
    )
