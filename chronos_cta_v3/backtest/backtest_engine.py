"""Backtesting utilities and walk-forward platform."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADE_COLUMNS = [
    "date",
    "action",
    "from_position",
    "to_position",
    "prediction",
    "price",
    "pnl",
    "equity",
]


def backtest(returns: pd.Series, positions: pd.Series) -> pd.Series:
    pnl = returns.fillna(0) * positions.shift(1).fillna(0)
    equity = (1 + pnl).cumprod()
    return equity


def auto_generate_strategy(predictions: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Auto strategy generation from alpha predictions."""
    pos = pd.Series(0, index=predictions.index, dtype=float)
    pos[predictions > threshold] = 1.0
    pos[predictions < -threshold] = -1.0
    return pos


def optimize_signal_threshold(predictions: pd.Series, returns: pd.Series, grid: list[float] | None = None) -> float:
    """Automatic parameter optimization via simple grid-search on Sharpe."""
    if grid is None:
        grid = [0.0, 0.0025, 0.005, 0.01, 0.02]

    best_thr = grid[0]
    best_score = -np.inf
    for thr in grid:
        pos = auto_generate_strategy(predictions, threshold=thr)
        pnl = returns.fillna(0) * pos.shift(1).fillna(0)
        std = pnl.std()
        score = 0 if std == 0 else np.sqrt(252) * pnl.mean() / std
        if score > best_score:
            best_score = score
            best_thr = thr
    return float(best_thr)


def walk_forward_backtest(
    predictions: pd.Series,
    returns: pd.Series,
    train_window: int = 120,
    test_window: int = 20,
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Walk-forward backtest with automatic threshold re-fit per window."""
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        mask = predictions.index >= start_ts
        predictions = predictions.loc[mask]
        returns = returns.loc[predictions.index]

    rows = []
    n = len(predictions)
    for start in range(train_window, n, test_window):
        train_slice = slice(start - train_window, start)
        test_slice = slice(start, min(start + test_window, n))

        thr = optimize_signal_threshold(predictions.iloc[train_slice], returns.iloc[train_slice])
        test_pred = predictions.iloc[test_slice]
        test_ret = returns.iloc[test_slice]
        pos = auto_generate_strategy(test_pred, threshold=thr)
        pnl = test_ret.fillna(0) * pos.shift(1).fillna(0)

        for i in range(len(test_pred)):
            rows.append(
                {
                    "date": test_pred.index[i],
                    "prediction": float(test_pred.iloc[i]),
                    "position": float(pos.iloc[i]),
                    "return": float(test_ret.iloc[i]),
                    "pnl": float(pnl.iloc[i]),
                    "threshold": float(thr),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["equity"] = (1 + out["pnl"]).cumprod()
    return out


def trade_records(daily: pd.DataFrame, prices: pd.Series | None = None) -> pd.DataFrame:
    """Generate trade records from a backtest daily dataframe."""
    if daily.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    records = []
    prev_pos = 0.0
    price_series = prices.reindex(daily["date"]).astype(float) if prices is not None else None

    for i, row in daily.iterrows():
        new_pos = float(row["position"])
        if np.isclose(new_pos, prev_pos):
            continue

        if np.isclose(prev_pos, 0.0) and not np.isclose(new_pos, 0.0):
            action = "OPEN"
        elif not np.isclose(prev_pos, 0.0) and np.isclose(new_pos, 0.0):
            action = "CLOSE"
        elif np.sign(prev_pos) != np.sign(new_pos):
            action = "REVERSE"
        else:
            action = "ADJUST"

        records.append(
            {
                "date": row["date"],
                "action": action,
                "from_position": prev_pos,
                "to_position": new_pos,
                "prediction": float(row["prediction"]),
                "price": float(price_series.iloc[i]) if price_series is not None and pd.notna(price_series.iloc[i]) else np.nan,
                "pnl": float(row["pnl"]),
                "equity": float(row.get("equity", np.nan)),
            }
        )
        prev_pos = new_pos

    return pd.DataFrame(records, columns=TRADE_COLUMNS)


@dataclass
class BacktestPlatform:
    """Complete backtest platform wrapper."""

    train_window: int = 120
    test_window: int = 20

    def run(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        start_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return walk_forward_backtest(
            predictions=predictions,
            returns=returns,
            train_window=self.train_window,
            test_window=self.test_window,
            start_date=start_date,
        )

    def run_with_trade_records(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        prices: pd.Series | None = None,
        start_date: str | pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        daily = self.run(predictions=predictions, returns=returns, start_date=start_date)
        return daily, trade_records(daily, prices=prices)
