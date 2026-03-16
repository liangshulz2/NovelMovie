"""Futures market data loader."""

import akshare as ak
import pandas as pd


def load_futures(symbol: str) -> pd.DataFrame:
    """Load Chinese futures daily bars from Sina via AkShare."""
    df = ak.futures_zh_daily_sina(symbol=symbol)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume", "hold"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["close"])
