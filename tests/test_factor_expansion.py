import numpy as np
import pandas as pd

from chronos_cta_v3.factors.factor_library import add_cross_sectional_factors, compute_factors


def test_new_timeseries_factors_exist():
    n = 140
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    base = pd.Series(100 + np.linspace(0, 20, n))
    raw = pd.DataFrame(
        {
            "date": dates,
            "open": base * (1 + 0.001),
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base,
            "volume": np.linspace(1000, 3000, n),
            "hold": np.linspace(500, 1500, n),
        }
    )

    out = compute_factors(raw)
    required = {
        "up_var_share_20",
        "jump_ratio_20",
        "range_to_close_vol_20",
        "breakout_persist_5",
        "trend_r2_20",
        "slope_over_noise_20",
        "price_volume_corr_20",
        "signed_volume_20",
        "oi_price_corr_20",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "body_ratio",
        "close_rank_20",
        "vol_rank_60",
        "oi_rank_60",
    }
    assert required.issubset(out.columns)


def test_cross_sectional_factors_exist_with_panel_input():
    n = 90
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rows = []
    for symbol, shift in [("RB0", 0.0), ("CU0", 3.0), ("M0", -2.0)]:
        close = 100 + shift + np.linspace(0, 10, n)
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": np.linspace(1000, 3000, n),
                    "hold": np.linspace(500, 1400, n),
                }
            )
        )
    panel = pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    fac = panel.groupby("symbol", group_keys=False).apply(compute_factors).reset_index(drop=True)
    fac = add_cross_sectional_factors(fac)

    required = {"cs_mom_rank", "cs_reversal_5", "cs_vol_rank", "cs_oi_chg_rank", "crowding_score", "industry_mom_rank"}
    assert required.issubset(fac.columns)
