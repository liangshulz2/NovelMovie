"""Risk budgeting / risk parity helper."""

import numpy as np
import pandas as pd


def inverse_vol_weights(vols: pd.Series, min_vol: float = 1e-6) -> pd.Series:
    clipped = vols.clip(lower=min_vol)
    inv = 1.0 / clipped
    return inv / inv.sum()


def apply_risk_budget(raw_positions: pd.Series, vols: pd.Series, max_portfolio_risk: float) -> pd.Series:
    total_risk = np.sum(np.abs(raw_positions.values) * vols.values)
    if total_risk <= max_portfolio_risk or total_risk <= 0:
        return raw_positions
    scale = max_portfolio_risk / total_risk
    return raw_positions * scale
