import pandas as pd

from chronos_cta_v3.factors.factor_library import compute_factors


def test_seasonality_features_use_only_history():
    dates = pd.date_range("2024-01-01", periods=13, freq="MS")
    close = pd.Series(range(100, 113), dtype=float)
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
        }
    )

    out = compute_factors(df)

    jan_first = out.loc[out["date"] == pd.Timestamp("2024-01-01"), "seasonality_month_mean"].iloc[0]
    jan_second = out.loc[out["date"] == pd.Timestamp("2025-01-01"), "seasonality_month_mean"].iloc[0]

    assert pd.isna(jan_first)
    assert pd.notna(jan_second)
