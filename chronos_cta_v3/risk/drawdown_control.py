"""Drawdown controls for strategy de-risking."""

import pandas as pd


def compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1


def drawdown_multiplier(equity: pd.Series, max_dd: float = -0.15) -> float:
    dd = compute_drawdown(equity).iloc[-1]
    if dd > max_dd:
        return 1.0
    return max(0.0, 1.0 - abs(dd - max_dd) * 5)
