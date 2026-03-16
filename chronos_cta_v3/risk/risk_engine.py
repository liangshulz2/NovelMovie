"""Portfolio level risk calculations."""

import numpy as np
import pandas as pd


def portfolio_risk(positions: pd.Series, vols: pd.Series) -> float:
    return float(np.sum(np.abs(positions.values) * vols.values))


def leverage(positions: pd.Series) -> float:
    return float(np.sum(np.abs(positions.values)))
