#!/usr/bin/env python3
"""One-click diagnosis for walk-forward short-signal behavior."""

from __future__ import annotations

import argparse

import numpy as np
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
    parser.add_argument("--min-short-days", type=int, default=8, help="Minimum short-signal days for whitelist.")
    parser.add_argument("--min-short-avg-pnl", type=float, default=0.0, help="Minimum average short directional pnl.")
    parser.add_argument("--min-short-win-rate", type=float, default=0.5, help="Minimum short win rate for whitelist.")
    parser.add_argument("--min-short-tstat", type=float, default=1.0, help="Minimum short t-stat for whitelist.")
    return parser.parse_args()


def _calc_t_stat(sum_ret: float, sum_sq_ret: float, n: int) -> float:
    if n <= 1:
        return float("nan")
    mean_ret = sum_ret / n
    var = (sum_sq_ret - n * mean_ret * mean_ret) / (n - 1)
    if var <= 0:
        return float("nan")
    std = float(np.sqrt(var))
    return float(mean_ret / (std / np.sqrt(n)))


def _short_significance_mark(short_days: int, short_avg_pnl: float, short_win_rate: float, short_t_stat: float) -> str:
    if short_days < 8:
        return "样本不足"
    if pd.notna(short_avg_pnl) and pd.notna(short_win_rate) and pd.notna(short_t_stat):
        if short_avg_pnl > 0 and short_win_rate >= 0.55 and short_t_stat >= 2.0:
            return "★★★"
        if short_avg_pnl > 0 and short_win_rate >= 0.5 and short_t_stat >= 1.0:
            return "★★"
        if short_avg_pnl > 0:
            return "★"
    return "✗"


def _build_short_whitelist_table(diag_by_symbol: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    for symbol, diag in diag_by_symbol.items():
        short_days = int(diag["short_count"].sum()) if "short_count" in diag else int(diag["neg_position_days"].sum())
        long_days = int(diag["long_count"].sum()) if "long_count" in diag else int(diag["pos_position_days"].sum())
        short_sum_ret = float(diag["short_sum_ret"].sum()) if "short_sum_ret" in diag else 0.0
        short_sum_sq_ret = float(diag["short_sum_sq_ret"].sum()) if "short_sum_sq_ret" in diag else 0.0
        short_win_days = int(diag["short_win_days"].sum()) if "short_win_days" in diag else 0

        short_avg_ret = short_sum_ret / short_days if short_days > 0 else float("nan")
        short_avg_pnl = -short_avg_ret if short_days > 0 else float("nan")
        short_win_rate = short_win_days / short_days if short_days > 0 else float("nan")
        short_t_stat = -_calc_t_stat(short_sum_ret, short_sum_sq_ret, short_days) if short_days > 1 else float("nan")
        mark = _short_significance_mark(short_days, short_avg_pnl, short_win_rate, short_t_stat)
        in_whitelist = (
            short_days >= args.min_short_days
            and pd.notna(short_avg_pnl)
            and short_avg_pnl >= args.min_short_avg_pnl
            and pd.notna(short_win_rate)
            and short_win_rate >= args.min_short_win_rate
            and pd.notna(short_t_stat)
            and short_t_stat >= args.min_short_tstat
        )

        rows.append(
            {
                "symbol": symbol,
                "short_days": short_days,
                "long_days": long_days,
                "short_avg_ret": short_avg_ret,
                "short_avg_pnl": short_avg_pnl,
                "short_win_rate": short_win_rate,
                "short_t_stat": short_t_stat,
                "significance_mark": mark,
                "short_whitelist": in_whitelist,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            by=["short_whitelist", "short_t_stat", "short_avg_pnl", "short_days"],
            ascending=[False, False, False, False],
        )
    return out


def main() -> None:
    args = parse_args()
    symbols = [args.symbol] if args.symbol else SYMBOLS
    diag_by_symbol: dict[str, pd.DataFrame] = {}

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
        diag_by_symbol[symbol] = diag

        total_neg_pred = int(diag["pred_lt_neg_thr"].sum())
        total_neg_pos = int(diag["neg_position_days"].sum())
        total_points = int(diag["test_points"].sum())
        short_days = int(diag["neg_position_days"].sum())
        long_days = int(diag["pos_position_days"].sum())
        short_avg = float(diag["short_avg_ret"].dropna().mean()) if "short_avg_ret" in diag else float("nan")
        long_avg = float(diag["long_avg_ret"].dropna().mean()) if "long_avg_ret" in diag else float("nan")
        short_avg_pnl = float(diag["short_avg_pnl"].dropna().mean()) if "short_avg_pnl" in diag else float("nan")
        long_avg_pnl = float(diag["long_avg_pnl"].dropna().mean()) if "long_avg_pnl" in diag else float("nan")
        short_hit = float(diag["short_win_rate"].dropna().mean()) if "short_win_rate" in diag else float("nan")
        long_hit = float(diag["long_win_rate"].dropna().mean()) if "long_win_rate" in diag else float("nan")
        print(
            f"\n[{symbol}] 窗口数={len(diag)}, 测试点={total_points}, "
            f"pred<-thr 合计={total_neg_pred}, -1持仓天数合计={total_neg_pos}, "
            f"short_days={short_days}, long_days={long_days}, "
            f"short_avg_ret={short_avg:.6f}, long_avg_ret={long_avg:.6f}, "
            f"short_avg_pnl={short_avg_pnl:.6f}, long_avg_pnl={long_avg_pnl:.6f}, "
            f"short_win_rate={short_hit:.2%}, long_win_rate={long_hit:.2%}"
        )
        print(diag.to_string(index=False))

    if diag_by_symbol:
        summary = _build_short_whitelist_table(diag_by_symbol, args)
        print("\n=== 可空品种白名单（含显著性标记）===")
        print(
            "过滤条件: "
            f"short_days>={args.min_short_days}, "
            f"short_avg_pnl>={args.min_short_avg_pnl}, "
            f"short_win_rate>={args.min_short_win_rate}, "
            f"short_t_stat>={args.min_short_tstat}"
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
