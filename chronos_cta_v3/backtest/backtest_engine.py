"""Backtesting utilities."""

import pandas as pd


def backtest(returns: pd.Series, positions: pd.Series) -> pd.Series:
    pnl = returns.fillna(0) * positions.shift(1).fillna(0)
    equity = (1 + pnl).cumprod()
    return equity
