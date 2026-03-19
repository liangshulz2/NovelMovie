"""Legacy wrapper around the unified module entrypoint.

Prefer running: `python -m chronos_cta_v3.main`
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from chronos_cta_v3.backtest.backtest_engine import BacktestPlatform
from chronos_cta_v3.config import (
    COMMISSION,
    MIN_HISTORY,
    NO_TRADE_BAND,
    POSITION_SMOOTHING,
    REGIME_EVENT_POSITION_SCALE,
    SLIPPAGE,
    SYMBOLS,
)
from chronos_cta_v3.data.futures_loader import load_futures


MIN_WINDOW = MIN_HISTORY


def _build_symbol_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Build prediction/return/price series for a single symbol."""
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0.0)
    # Use lagged short-term momentum as a lightweight daily prediction proxy.
    pred = ret.rolling(5, min_periods=3).mean().shift(1).fillna(0.0)
    return pred, ret, close


def _performance_stats(equity: pd.Series) -> dict:
    if equity.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    ret = equity.pct_change().dropna()
    if ret.empty:
        return {
            "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    n = len(ret)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    annual_return = float((1 + total_return) ** (252 / max(n, 1)) - 1)
    annual_vol = float(ret.std() * np.sqrt(252))
    sharpe = float(0.0 if annual_vol <= 1e-12 else annual_return / annual_vol)
    max_drawdown = float((equity / equity.cummax() - 1).min())

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def run_portfolio_backtest(
    initial_capital: float = 100000,
    start_date: str = "2025-08-01",
    end_date: str = "2026-03-18",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a date-ranged portfolio backtest and output performance metrics.

    Returns:
        portfolio_daily: Portfolio-level daily pnl/equity series.
        symbol_daily: Per-symbol daily backtest details.
        trades: Portfolio-level trade records.
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if start_ts > end_ts:
        raise ValueError(f"start_date({start_date}) cannot be after end_date({end_date}).")

    bt = BacktestPlatform(
        train_window=120,
        test_window=20,
        commission=COMMISSION,
        slippage=SLIPPAGE,
        no_trade_band=NO_TRADE_BAND,
        position_smoothing=POSITION_SMOOTHING,
        regime_event_position_scale=REGIME_EVENT_POSITION_SCALE,
    )

    symbol_frames: list[pd.DataFrame] = []
    port_pred_parts: list[pd.Series] = []
    port_ret_parts: list[pd.Series] = []
    port_price_parts: list[pd.Series] = []

    for symbol in SYMBOLS:
        raw = load_futures(symbol)
        if raw.empty or "close" not in raw.columns:
            continue
        df = raw.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        if not isinstance(df.index, pd.DatetimeIndex):
            continue
        df = df.sort_index()
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        if len(df) < max(bt.train_window + bt.test_window, 40):
            continue

        pred, ret, close = _build_symbol_series(df)
        daily, _ = bt.run_with_trade_records(predictions=pred, returns=ret, prices=close, start_date=start_ts)
        if daily.empty:
            continue

        daily = daily.copy()
        daily["symbol"] = symbol
        symbol_frames.append(daily)

        aligned = daily.set_index("date")
        port_pred_parts.append(aligned["prediction"].rename(symbol))
        port_ret_parts.append(aligned["return"].rename(symbol))
        port_price_parts.append(close.reindex(aligned.index).rename(symbol))

    if not symbol_frames:
        empty = pd.DataFrame()
        return empty, empty, empty

    symbol_daily = pd.concat(symbol_frames, ignore_index=True).sort_values(["date", "symbol"])

    portfolio_pred = pd.concat(port_pred_parts, axis=1).mean(axis=1).sort_index()
    portfolio_ret = pd.concat(port_ret_parts, axis=1).mean(axis=1).sort_index()
    portfolio_price = pd.concat(port_price_parts, axis=1).mean(axis=1).sort_index()

    portfolio_daily, trades = bt.run_with_trade_records(
        predictions=portfolio_pred,
        returns=portfolio_ret,
        prices=portfolio_price,
        start_date=start_ts,
    )
    if portfolio_daily.empty:
        return portfolio_daily, symbol_daily, trades

    portfolio_daily = portfolio_daily.copy()
    portfolio_daily["capital"] = initial_capital * portfolio_daily["equity"]

    stats = _performance_stats(portfolio_daily["equity"])
    for k, v in stats.items():
        portfolio_daily[k] = v

    return portfolio_daily, symbol_daily, trades


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
        latest = portfolio_daily.iloc[-1]
        print(f"组合回测资金：{args.initial_capital:,.2f} 元")
        print(f"回测区间：{args.start_date} ~ {args.end_date}")
        print(
            "回测统计: "
            f"总收益={latest['total_return']:.2%}, "
            f"年化收益={latest['annual_return']:.2%}, "
            f"最大回撤={latest['max_drawdown']:.2%}, "
            f"夏普比率={latest['sharpe']:.3f}"
        )
        print("\n组合净值（前20行）：")
        print(portfolio_daily.head(20).to_string(index=False))
        print(f"\n品种明细行数: {len(symbol_daily)}")
        print(f"交易记录条数: {len(trades)}")
