import pandas as pd

from chronos_cta_v3.backtest.backtest_engine import backtest, trade_records


def test_backtest_uses_shifted_positions():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.Series([0.1, 0.1, 0.0], index=idx)
    positions = pd.Series([1.0, 0.0, 0.0], index=idx)

    equity = backtest(returns, positions)

    # Day1 uses shifted(1)=0 so pnl=0; day2 uses previous pos=1 so pnl=0.1
    assert equity.iloc[0] == 1.0
    assert abs(equity.iloc[1] - 1.1) < 1e-12


def test_trade_records_empty_schema():
    daily = pd.DataFrame()
    trades = trade_records(daily)
    assert trades.empty
    assert trades.columns.tolist() == [
        "date",
        "action",
        "from_position",
        "to_position",
        "prediction",
        "price",
        "pnl",
        "equity",
    ]
