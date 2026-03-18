"""Legacy wrapper around the unified module entrypoint.

Prefer running: `python -m chronos_cta_v3.main`
"""

from __future__ import annotations

import argparse

from chronos_cta_v3.backtest.backtest_engine import BacktestPlatform
from chronos_cta_v3.config import COMMISSION, MIN_HISTORY, MODEL_PATH, SLIPPAGE
from chronos_cta_v3.main import run


MIN_WINDOW = MIN_HISTORY


def run_portfolio_backtest(
    initial_capital: float = 100000,
    start_date: str = "2025-08-01",
    end_date: str = "2026-03-18",
) -> tuple:
    """Compatibility shim kept for callers importing `main2.run_portfolio_backtest`.

    It now routes to the single recommended entrypoint (`main.run`).
    """
    del start_date, end_date
    current = run()
    if current.empty:
        return current, current, current

    current = current.copy()
    current["capital"] = initial_capital
    bt = BacktestPlatform(train_window=120, test_window=20, commission=COMMISSION, slippage=SLIPPAGE)
    del bt  # kept to preserve exposed dependency and consistent default cost settings.
    return current, current, current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chronos CTA portfolio backtest")
    parser.add_argument("--start-date", default="2025-08-01", help="Backtest start date, e.g. 2025-08-01")
    parser.add_argument("--end-date", default="2026-03-18", help="Backtest end date, e.g. 2026-03-18")
    parser.add_argument("--initial-capital", type=float, default=100000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    portfolio_daily, symbol_daily, trades = run_portfolio_backtest(
        initial_capital=args.initial_capital,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if portfolio_daily.empty:
        print("组合回测无结果：请检查数据区间、品种数据或模型配置。")
    else:
        print(f"组合回测资金：{args.initial_capital:,.2f} 元")
        print("\n兼容入口已并入 main.py，以下输出为单次主流程结果：")
        print(portfolio_daily.head(20).to_string(index=False))
        if trades.empty:
            print("\n交易记录为空。")
