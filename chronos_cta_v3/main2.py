import pandas as pd
from data.futures_loader import load_futures
from model.chronos_model import ChronosModel
from model.xgb_model import XGBModel
from model.ensemble_model import bayesian_model_averaging
from alpha.alpha_engine import generate_signal
from backtest.backtest_engine import BacktestPlatform
from factors.factor_library import compute_factors
from factors.factor_selector import select_features

# ------------------- 参数 -------------------
SYMBOLS = [
    "RB0",
    "SA0",
    "FG0",
    "CU0",
    "AL0",
    "M0",
    "Y0",
]
start_date = "2025-08-01"
end_date = "2026-03-18"
MODEL_PATH = "E:/ai/chronos-2"
DEVICE = "cpu"
INITIAL_CAPITAL = 100000  # 可修改，例如 10 万元
MIN_WINDOW = 21


def generate_symbol_signals(
    symbol: str,
    chronos: ChronosModel,
    xgb: XGBModel,
    start: str,
    end: str,
    min_window: int,
) -> pd.DataFrame:
    """为单个期货品种生成信号与收益序列。"""
    df = load_futures(symbol)
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    if len(df) <= min_window:
        return pd.DataFrame(columns=["date", "signal", "ret", "close", "symbol"])

    signals = []
    aligned_dates = []

    for i in range(min_window, len(df)):
        hist = df.iloc[:i].copy()
        hist = compute_factors(hist)
        hist["target"] = hist["close"].pct_change().shift(-1)

        clean_hist = hist.dropna(subset=["target"])
        if len(clean_hist) < 2:
            continue

        feats = select_features(clean_hist, "target", top_n=30)
        if not feats:
            feats = ["close", "open", "high", "low", "volume"]

        model_df = clean_hist[[*feats, "target"]].dropna()
        if len(model_df) < 2:
            continue

        x_train = model_df[feats].iloc[:-1]
        y_train = model_df["target"].iloc[:-1]
        x_test = model_df[feats].iloc[[-1]]

        xgb.fit(x_train.values, y_train.values)
        ml_pred = float(xgb.predict(x_test.values)[0])

        close = hist["close"].dropna()
        factor_mat = hist[feats].fillna(0)
        chronos_pred = chronos.predict(close, factor_mat, 1)
        chronos_alpha = (
            float(chronos_pred[0])
            if hasattr(chronos_pred, "__getitem__")
            else float(chronos_pred)
        )

        target_last = model_df["target"].iloc[-1]
        alpha, _ = bayesian_model_averaging(
            predictions={"chronos": chronos_alpha, "xgb": ml_pred},
            errors={
                "chronos": abs(float(target_last - chronos_alpha)) + 1e-6,
                "xgb": abs(float(target_last - ml_pred)) + 1e-6,
            },
        )

        signals.append(generate_signal(alpha))
        aligned_dates.append(df.loc[i, "date"])

    if not signals:
        return pd.DataFrame(columns=["date", "signal", "ret", "close", "symbol"])

    out = (
        df.loc[df["date"].isin(aligned_dates), ["date", "close"]]
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


def run_portfolio_backtest(initial_capital: float = INITIAL_CAPITAL) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    # 组合层：按日期把各品种 PnL 做等权平均
    portfolio_daily = (
        all_daily.groupby("date", as_index=False)
        .agg(
            symbols=("symbol", "nunique"),
            avg_pnl=("pnl", "mean"),
            avg_position=("position", "mean"),
        )
        .sort_values("date")
    )
    portfolio_daily["equity"] = (1 + portfolio_daily["avg_pnl"]).cumprod()
    portfolio_daily["capital"] = initial_capital * portfolio_daily["equity"]
    portfolio_daily["daily_pnl_amount"] = portfolio_daily["capital"].shift(1).fillna(initial_capital) * portfolio_daily["avg_pnl"]

    trades_all = pd.concat(trade_records_list, ignore_index=True) if trade_records_list else pd.DataFrame()
    return portfolio_daily, all_daily, trades_all


if __name__ == "__main__":
    portfolio_daily, symbol_daily, trades = run_portfolio_backtest(initial_capital=INITIAL_CAPITAL)

    if portfolio_daily.empty:
        print("组合回测无结果：请检查数据区间、品种数据或模型配置。")
    else:
        print(f"组合回测资金：{INITIAL_CAPITAL:,.2f} 元")
        print("\n组合每日结果（前 20 行）：")
        print(portfolio_daily.head(20).to_string(index=False))

        print("\n分品种每日结果（前 20 行）：")
        print(symbol_daily.head(20).to_string(index=False))

        if trades.empty:
            print("\n交易记录为空。")
        else:
            print("\n交易记录（前 20 行）：")
            print(trades.head(20).to_string(index=False))
