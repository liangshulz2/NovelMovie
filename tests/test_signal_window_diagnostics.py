import pandas as pd

from chronos_cta_v3.backtest.backtest_engine import diagnose_signal_windows


def test_diagnose_signal_windows_counts_and_threshold() -> None:
    idx = pd.date_range("2026-01-01", periods=8, freq="D")
    predictions = pd.Series([0.01, 0.01, 0.01, 0.01, -0.2, 0.2, -0.1, 0.0], index=idx)
    returns = pd.Series(0.0, index=idx)

    out = diagnose_signal_windows(predictions, returns, train_window=4, test_window=4)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["threshold"] == 0.0
    assert row["test_points"] == 4
    assert row["pred_lt_neg_thr"] == 2
    assert row["pred_gt_pos_thr"] == 1
    assert row["neg_position_days"] == 2
    assert row["pos_position_days"] == 1
    assert row["flat_days"] == 1
