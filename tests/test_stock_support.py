from pathlib import Path

import numpy as np
import pandas as pd

from chronos_cta_v3.data.stock_loader import load_stock_from_csv, load_stock_list
from chronos_cta_v3.factors.factor_library import compute_factors


def test_stock_loader_normalizes_cn_columns(tmp_path: Path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    list_file = tmp_path / "stocks.txt"
    list_file.write_text("000001\n", encoding="utf-8")

    raw = pd.DataFrame(
        {
            "日期": pd.date_range("2024-01-01", periods=5, freq="B"),
            "开盘": [10, 10.2, 10.1, 10.4, 10.5],
            "最高": [10.3, 10.4, 10.5, 10.6, 10.8],
            "最低": [9.9, 10.0, 9.95, 10.2, 10.3],
            "收盘": [10.1, 10.25, 10.2, 10.5, 10.6],
            "成交量": [1000, 1200, 1100, 1500, 1600],
            "成交额": [10000, 12300, 11220, 15750, 16960],
            "换手率": [1.0, 1.1, 1.05, 1.2, 1.3],
        }
    )
    raw.to_csv(csv_dir / "000001.csv", index=False)

    codes = load_stock_list(list_file)
    assert codes == ["000001"]
    out = load_stock_from_csv("000001", csv_dir)
    assert {"date", "open", "high", "low", "close", "volume", "amount", "turnover_rate", "symbol"}.issubset(out.columns)
    assert out["symbol"].iloc[0] == "000001"


def test_stock_specific_factors_exist():
    n = 140
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(20 + np.linspace(0, 3, n) + np.sin(np.linspace(0, 8, n)) * 0.2)
    high = close * (1 + 0.015)
    low = close * (1 - 0.015)
    out = compute_factors(
        pd.DataFrame(
            {
                "date": dates,
                "open": close * (1 - 0.002),
                "high": high,
                "low": low,
                "close": close,
                "volume": np.linspace(5e6, 8e6, n),
                "amount": close * np.linspace(5e6, 8e6, n),
                "turnover_rate": np.linspace(0.8, 2.2, n),
            }
        )
    )

    required = {
        "amount_chg_5",
        "amount_ratio_20_60",
        "avg_trade_price",
        "avg_trade_price_gap",
        "money_flow_multiplier",
        "money_flow_volume_20",
        "adl_slope_20",
        "chaikin_osc",
        "turnover_rate_5",
        "turnover_rate_20",
        "turnover_vol_adj_20",
    }
    assert required.issubset(out.columns)
