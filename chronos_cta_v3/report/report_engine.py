"""Performance report generation."""

import numpy as np
import pandas as pd


def performance_stats(equity: pd.Series) -> dict:
    ret = equity.pct_change().dropna()
    if ret.empty:
        return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

    years = len(ret) / 252
    cagr = equity.iloc[-1] ** (1 / max(years, 1e-6)) - 1
    sharpe = np.sqrt(252) * ret.mean() / max(ret.std(), 1e-8)
    dd = (equity / equity.cummax() - 1).min()
    return {"cagr": float(cagr), "sharpe": float(sharpe), "max_drawdown": float(dd)}
