import pandas as pd

from chronos_cta_v3.backtest.backtest_engine import backtest, trade_records, walk_forward_backtest


def test_backtest_uses_shifted_positions():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.Series([0.1, 0.1, 0.0], index=idx)
    positions = pd.Series([1.0, 0.0, 0.0], index=idx)

    equity = backtest(returns, positions)

    # Day1 uses shifted(1)=0 so pnl=0; day2 uses previous pos=1 so pnl=0.1
    assert equity.iloc[0] == 1.0
    assert abs(equity.iloc[1] - 1.1) < 1e-12


def test_backtest_costs_reduce_equity():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.Series([0.0, 0.0, 0.0], index=idx)
    positions = pd.Series([1.0, -1.0, 0.0], index=idx)

    equity_no_cost = backtest(returns, positions, commission=0.0, slippage=0.0)
    equity_with_cost = backtest(returns, positions, commission=0.001, slippage=0.001)

    assert equity_no_cost.iloc[-1] == 1.0
    assert equity_with_cost.iloc[-1] < equity_no_cost.iloc[-1]


def test_walk_forward_carries_prev_position_across_windows():
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    predictions = pd.Series([0.2] * len(idx), index=idx)
    returns = pd.Series([0.01] * len(idx), index=idx)

    daily = walk_forward_backtest(
        predictions=predictions,
        returns=returns,
        train_window=2,
        test_window=2,
        commission=0.0,
        slippage=0.0,
    )

    # The first day in the 2nd test window should inherit previous window position=1.
    assert daily.iloc[2]["prev_position"] == 1.0


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
        "exec_price",
        "price_slippage",
        "transaction_cost",
        "pnl",
        "equity",
    ]
