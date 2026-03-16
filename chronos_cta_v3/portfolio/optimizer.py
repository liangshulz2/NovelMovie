"""Portfolio optimizer utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .risk_parity import risk_parity_weights


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr(numeric_only=True)


def optimize_portfolio(
    returns: pd.DataFrame,
    method: str = "risk_parity",
    risk_aversion: float = 3.0,
) -> pd.Series:
    """Simple portfolio optimizer supporting risk parity and mean-variance."""
    if returns.empty:
        return pd.Series(dtype=float)

    if method == "risk_parity":
        cov = returns.cov(numeric_only=True).fillna(0)
        return risk_parity_weights(cov)

    mu = returns.mean(numeric_only=True).values
    cov = returns.cov(numeric_only=True).values
    if cov.size == 0:
        return pd.Series(dtype=float)

    inv = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))
    w = inv @ mu / max(risk_aversion, 1e-6)
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    return pd.Series(w, index=returns.columns)
