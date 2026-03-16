"""Risk budgeting / risk parity helper."""

from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_vol_weights(vols: pd.Series, min_vol: float = 1e-6) -> pd.Series:
    clipped = vols.clip(lower=min_vol)
    inv = 1.0 / clipped
    return inv / inv.sum()


def risk_parity_weights(cov: pd.DataFrame, max_iter: int = 500, tol: float = 1e-6) -> pd.Series:
    """Iterative risk parity optimizer from covariance matrix."""
    n = len(cov)
    w = np.ones(n) / n
    cov_m = cov.values

    for _ in range(max_iter):
        port_var = float(w.T @ cov_m @ w)
        if port_var <= 0:
            break
        mrc = cov_m @ w
        rc = w * mrc / np.sqrt(port_var)
        target = np.mean(rc)

        prev = w.copy()
        for i in range(n):
            if mrc[i] != 0:
                w[i] = w[i] * target / max(rc[i], 1e-10)
        w = np.clip(w, 1e-8, None)
        w = w / w.sum()

        if np.max(np.abs(w - prev)) < tol:
            break

    return pd.Series(w, index=cov.index)


def apply_risk_budget(raw_positions: pd.Series, vols: pd.Series, max_portfolio_risk: float) -> pd.Series:
    total_risk = np.sum(np.abs(raw_positions.values) * vols.values)
    if total_risk <= max_portfolio_risk or total_risk <= 0:
        return raw_positions
    scale = max_portfolio_risk / total_risk
    return raw_positions * scale
