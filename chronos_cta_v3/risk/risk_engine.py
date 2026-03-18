"""Portfolio level risk calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_risk(
    positions: pd.Series,
    vols: pd.Series | None = None,
    cov_matrix: pd.DataFrame | None = None,
) -> float:
    weights = positions.astype(float).fillna(0.0)
    if cov_matrix is not None and not cov_matrix.empty:
        cov = cov_matrix.reindex(index=weights.index, columns=weights.index).fillna(0.0)
        return float(np.sqrt(np.maximum(weights.values @ cov.values @ weights.values, 0.0)))

    if vols is None:
        return 0.0
    return float(np.sum(np.abs(weights.values) * vols.reindex(weights.index).fillna(0.0).values))


def leverage(positions: pd.Series) -> float:
    return float(np.sum(np.abs(positions.values)))
