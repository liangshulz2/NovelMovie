import pandas as pd

from chronos_cta_v3.factor_eval import EvalConfig, evaluate_factors


def test_evaluate_factors_with_empty_factor_list_returns_empty_summary_with_columns():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "symbol": ["RB0"] * 5,
            "close": [100, 101, 102, 103, 104],
        }
    )

    summary, qret_dict, ic_dict = evaluate_factors(df, [], EvalConfig())

    assert summary.empty
    assert "recommendation" in summary.columns
    assert qret_dict == {}
    assert ic_dict == {}


def test_evaluate_factors_skips_missing_factor_columns_without_crashing():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "symbol": ["RB0"] * 5,
            "close": [100, 101, 102, 103, 104],
        }
    )

    summary, qret_dict, ic_dict = evaluate_factors(df, ["f_missing"], EvalConfig())

    assert summary.empty
    assert "recommendation" in summary.columns
    assert qret_dict == {}
    assert ic_dict == {}
