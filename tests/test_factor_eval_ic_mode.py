import numpy as np
import pandas as pd

from chronos_cta_v3.factor_eval import EvalConfig, evaluate_factors


def test_auto_mode_falls_back_to_timeseries_ic_for_single_symbol_panel():
    n = 120
    close = np.linspace(100, 130, n)
    f1 = np.sin(np.linspace(0, 6, n))
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "symbol": ["RB0"] * n,
            "close": close,
            "f1": f1,
        }
    )

    cfg = EvalConfig()
    cfg.ic_mode = "auto"
    cfg.min_cs_size = 20
    cfg.min_ts_size = 30

    summary, _, _ = evaluate_factors(df, ["f1"], cfg)

    assert len(summary) == 1
    assert summary.loc[0, "ic_n"] > 0
    assert summary.loc[0, "recommendation"] != "INSUFFICIENT_DATA"


def test_insufficient_data_label_when_no_valid_ic_samples():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "symbol": ["RB0"] * 8,
            "close": np.linspace(100, 104, 8),
            "f1": np.ones(8),
        }
    )

    cfg = EvalConfig()
    cfg.ic_mode = "auto"
    cfg.min_ts_size = 20

    summary, _, _ = evaluate_factors(df, ["f1"], cfg)

    assert len(summary) == 1
    assert summary.loc[0, "ic_n"] == 0
    assert summary.loc[0, "recommendation"] == "INSUFFICIENT_DATA"
