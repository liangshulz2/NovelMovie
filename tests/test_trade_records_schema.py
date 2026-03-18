import pandas as pd

from chronos_cta_v3.backtest.backtest_engine import trade_records


def test_trade_records_schema_and_actions():
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "position": [0.0, 1.0, -1.0, 0.0],
            "prediction": [0.0, 0.5, -0.3, 0.1],
            "pnl": [0.0, 0.01, -0.02, 0.0],
            "equity": [1.0, 1.01, 0.9898, 0.9898],
        }
    )
    prices = pd.Series([100, 101, 99, 100], index=daily["date"])

    trades = trade_records(daily, prices)

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
    assert trades["action"].tolist() == ["OPEN", "REVERSE", "CLOSE"]
