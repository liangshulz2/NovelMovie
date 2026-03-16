"""Simple local cache utilities for dataframes."""

from pathlib import Path
import pandas as pd


class DataCache:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}.parquet"

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        df.to_parquet(self.path_for(symbol), index=False)

    def load(self, symbol: str) -> pd.DataFrame | None:
        path = self.path_for(symbol)
        if not path.exists():
            return None
        return pd.read_parquet(path)
