import numpy as np
import pandas as pd

from chronos_cta_v3.backtest.backtest_engine import walk_forward_backtest
from chronos_cta_v3.factors.factor_library import compute_factors


def test_data_factor_backtest_smoke():
    rng = np.random.default_rng(42)
    n = 80
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rets = rng.normal(0, 0.01, n)
    close = 100 * (1 + pd.Series(rets)).cumprod()

    raw = pd.DataFrame(
        {
            "date": dates,
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.integers(100, 1000, n),
        }
    )

    feat = compute_factors(raw)
    pred = feat["mom5"].fillna(0)
    ret = feat["close"].pct_change().fillna(0)

    daily = walk_forward_backtest(pred, ret, train_window=20, test_window=10, commission=0.0002, slippage=0.0003)

    assert not daily.empty
    assert {"pnl", "equity", "cost", "prev_position"}.issubset(daily.columns)
