import pandas as pd

from chronos_cta_v3.portfolio.risk_parity import apply_risk_budget
from chronos_cta_v3.risk.risk_engine import portfolio_risk


def test_risk_budget_scales_positions_monotonically():
    pos = pd.Series({"A": 1.0, "B": -1.0})
    vols = pd.Series({"A": 0.2, "B": 0.3})

    loose = apply_risk_budget(pos, vols, max_portfolio_risk=1.0)
    tight = apply_risk_budget(pos, vols, max_portfolio_risk=0.2)

    assert tight.abs().sum() < loose.abs().sum()
    assert portfolio_risk(tight, vols=vols) <= 0.2 + 1e-12
