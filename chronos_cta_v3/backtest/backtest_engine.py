"""Backtesting utilities and walk-forward platform."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
) -> pd.DataFrame:
    """Walk-forward backtest with automatic threshold re-fit per window."""
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


@dataclass
class BacktestPlatform:
    """Complete backtest platform wrapper."""

    train_window: int = 120
    test_window: int = 20

    def run(self, predictions: pd.Series, returns: pd.Series) -> pd.DataFrame:
        return walk_forward_backtest(
            predictions=predictions,
            returns=returns,
            train_window=self.train_window,
            test_window=self.test_window,
        )
