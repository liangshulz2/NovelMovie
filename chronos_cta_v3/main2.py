from __future__ import annotations

import argparse
import os

import pandas as pd

from alpha.alpha_engine import generate_signal
from backtest.backtest_engine import BacktestPlatform
from data.futures_loader import load_futures
from factors.factor_library import compute_factors
from factors.factor_selector import select_features
from model.chronos_model import ChronosModel
from model.ensemble_model import bayesian_model_averaging
from model.xgb_model import XGBModel

SYMBOLS = ["RB0", "SA0", "FG0", "CU0", "AL0", "M0", "Y0"]
MODEL_PATH = os.getenv("CHRONOS_MODEL_PATH", "E:/ai/chronos-2")
DEVICE = os.getenv("CHRONOS_DEVICE", "cpu")
INITIAL_CAPITAL = 100000
MIN_WINDOW = 21


def _compute_chronos_alpha(pred) -> float:
    arr = pd.Series(pred).astype(float).to_numpy().reshape(-1)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return float(arr[0])
    first = float(arr[0])
    return 0.0 if abs(first) < 1e-8 else float((arr.mean() - first) / abs(first))


def generate_symbol_signals(
    symbol: str,
    chronos: ChronosModel,
    xgb: XGBModel,
    start: str,
    end: str,
    min_window: int,
    retrain_every: int = 5,
    refit_features_every: int = 10,
) -> pd.DataFrame:
    """为单个期货品种生成信号与收益序列。"""
    df = load_futures(symbol, start_date=start, end_date=end)
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    if len(df) <= min_window:
        return pd.DataFrame(columns=["date", "signal", "ret", "close", "symbol"])

    factor_df = compute_factors(df.copy())
    factor_df["target"] = factor_df["close"].pct_change().shift(-1)

    signals = []
    aligned_dates = []
    feats: list[str] = []

    for i in range(min_window, len(factor_df)):
        hist = factor_df.iloc[:i].copy()
        clean_hist = hist.dropna(subset=["target"])
        if len(clean_hist) < 2:
            continue

        if (not feats) or ((i - min_window) % refit_features_every == 0):
            feats = select_features(clean_hist, "target", top_n=30)
            if not feats:
                feats = ["close", "open", "high", "low", "volume"]

        model_df = clean_hist[[*feats, "target"]].dropna()
        if len(model_df) < 2:
            continue

        if (i - min_window) % retrain_every == 0:
            x_train = model_df[feats].iloc[:-1]
            y_train = model_df["target"].iloc[:-1]
            xgb.fit(x_train.values, y_train.values)

        x_test = model_df[feats].iloc[[-1]]
        ml_pred = float(xgb.predict(x_test.values)[0])

        close = hist["close"].dropna()
        factor_mat = hist[feats].fillna(0)
        chronos_pred = chronos.predict(close, factor_mat, 1)
        chronos_alpha = _compute_chronos_alpha(chronos_pred)

        target_last = float(model_df["target"].iloc[-1])
        alpha, _ = bayesian_model_averaging(
            predictions={"chronos": chronos_alpha, "xgb": ml_pred},
            errors={
                "chronos": abs(target_last - chronos_alpha) + 1e-6,
                "xgb": abs(target_last - ml_pred) + 1e-6,
            },
        )

        signals.append(generate_signal(alpha))
        aligned_dates.append(factor_df.loc[i, "date"])

    if not signals:
        return pd.DataFrame(columns=["date", "signal", "ret", "close", "symbol"])

    out = (
        factor_df.loc[factor_df["date"].isin(aligned_dates), ["date", "close"]]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )
    out["signal"] = signals
    out["ret"] = out["close"].pct_change().fillna(0)
    out["symbol"] = symbol

    out["date"] = pd.to_datetime(out["date"])
    out = out.set_index("date").sort_index()
    return out


def run_portfolio_backtest(
    initial_capital: float = INITIAL_CAPITAL,
    start_date: str = "2025-08-01",
    end_date: str = "2026-03-18",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """组合回测：等权聚合每个品种的日度 PnL。"""
    chronos = ChronosModel(MODEL_PATH, DEVICE)
    xgb = XGBModel()
    bt = BacktestPlatform(train_window=120, test_window=20)

    symbol_daily = []
    trade_records_list = []

    for symbol in SYMBOLS:
        symbol_df = generate_symbol_signals(symbol, chronos, xgb, start_date, end_date, MIN_WINDOW)
        if symbol_df.empty:
            print(f"[{symbol}] 数据不足，跳过。")
            continue

        daily, trades = bt.run_with_trade_records(
            predictions=symbol_df["signal"],
            returns=symbol_df["ret"],
            prices=symbol_df["close"],
        )
        if daily.empty:
            print(f"[{symbol}] 回测结果为空，跳过。")
            continue

        daily["symbol"] = symbol
        symbol_daily.append(daily)

        if not trades.empty:
            trades["symbol"] = symbol
            trade_records_list.append(trades)

    if not symbol_daily:
        empty = pd.DataFrame()
        return empty, empty, empty

    all_daily = pd.concat(symbol_daily, ignore_index=True)

    portfolio_daily = (
        all_daily.groupby("date", as_index=False)
        .agg(symbols=("symbol", "nunique"), avg_pnl=("pnl", "mean"), avg_position=("position", "mean"))
        .sort_values("date")
    )
    portfolio_daily["equity"] = (1 + portfolio_daily["avg_pnl"]).cumprod()
    portfolio_daily["capital"] = initial_capital * portfolio_daily["equity"]
    portfolio_daily["daily_pnl_amount"] = portfolio_daily["capital"].shift(1).fillna(initial_capital) * portfolio_daily["avg_pnl"]

    trades_all = pd.concat(trade_records_list, ignore_index=True) if trade_records_list else pd.DataFrame()
    return portfolio_daily, all_daily, trades_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chronos CTA portfolio backtest")
    parser.add_argument("--start-date", default="2025-08-01", help="Backtest start date, e.g. 2025-08-01")
    parser.add_argument("--end-date", default="2026-03-18", help="Backtest end date, e.g. 2026-03-18")
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
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
        print("\n组合每日结果（前 20 行）：")
        print(portfolio_daily.head(20).to_string(index=False))

        print("\n分品种每日结果（前 20 行）：")
        print(symbol_daily.head(20).to_string(index=False))

        if trades.empty:
            print("\n交易记录为空。")
        else:
            print("\n交易记录（前 20 行）：")
            print(trades.head(20).to_string(index=False))
