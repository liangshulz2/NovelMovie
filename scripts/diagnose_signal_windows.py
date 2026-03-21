#!/usr/bin/env python3
"""One-click diagnosis for walk-forward short-signal behavior."""

from __future__ import annotations

import argparse

import pandas as pd

from chronos_cta_v3.backtest.backtest_engine import diagnose_signal_windows
from chronos_cta_v3.config import SYMBOLS
from chronos_cta_v3.data.futures_loader import load_futures
from chronos_cta_v3.main2 import _build_symbol_series


def _load_symbol_df(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    raw = load_futures(symbol)
    if raw.empty or "close" not in raw.columns:
        return pd.DataFrame()

    df = raw.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return df.sort_index().loc[(df.index >= start_ts) & (df.index <= end_ts)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose walk-forward thresholds and short-signal counts.")
    parser.add_argument("--symbol", default="", help="Single symbol, e.g. RB0. Empty means all configured symbols.")
    parser.add_argument("--start-date", default="2025-08-01")
    parser.add_argument("--end-date", default="2026-03-18")
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--test-window", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [args.symbol] if args.symbol else SYMBOLS

    for symbol in symbols:
        df = _load_symbol_df(symbol, args.start_date, args.end_date)
        if df.empty:
            print(f"[{symbol}] 无可用数据，跳过。")
            continue

        pred, ret, _ = _build_symbol_series(df)
        diag = diagnose_signal_windows(
            predictions=pred,
            returns=ret,
            train_window=args.train_window,
            test_window=args.test_window,
            start_date=args.start_date,
        )
        if diag.empty:
            print(f"[{symbol}] 诊断结果为空（样本长度可能不足 train_window+test_window）。")
            continue

        total_neg_pred = int(diag["pred_lt_neg_thr"].sum())
        total_neg_pos = int(diag["neg_position_days"].sum())
        total_points = int(diag["test_points"].sum())
        short_days = int(diag["neg_position_days"].sum())
        long_days = int(diag["pos_position_days"].sum())
        short_avg = float(diag["short_avg_ret"].dropna().mean()) if "short_avg_ret" in diag else float("nan")
        long_avg = float(diag["long_avg_ret"].dropna().mean()) if "long_avg_ret" in diag else float("nan")
        short_hit = float(diag["short_win_rate"].dropna().mean()) if "short_win_rate" in diag else float("nan")
        long_hit = float(diag["long_win_rate"].dropna().mean()) if "long_win_rate" in diag else float("nan")
        print(
            f"\n[{symbol}] 窗口数={len(diag)}, 测试点={total_points}, "
            f"pred<-thr 合计={total_neg_pred}, -1持仓天数合计={total_neg_pos}, "
            f"short_days={short_days}, long_days={long_days}, "
            f"short_avg_ret={short_avg:.6f}, long_avg_ret={long_avg:.6f}, "
            f"short_win_rate={short_hit:.2%}, long_win_rate={long_hit:.2%}"
        )
        print(diag.to_string(index=False))


if __name__ == "__main__":
    main()
