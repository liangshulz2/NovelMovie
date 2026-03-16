"""CTA factor library (80+ factors) for cross-commodity research."""

import numpy as np
import pandas as pd


ROLL_WINDOWS = [3, 5, 10, 20, 30, 60]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"].astype(float)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    open_ = d["open"].astype(float)
    volm = d.get("volume", pd.Series(np.nan, index=d.index)).astype(float)

    ret1 = close.pct_change()
    log_ret = np.log(close).diff()
    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    # Momentum family (15)
    for w in [2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 90, 120]:
        d[f"mom{w}"] = close.pct_change(w)
    d["mom_sign20"] = np.sign(d["mom20"])
    d["mom_accel"] = d["mom10"] - d["mom30"]
    d["mom_curve"] = d["mom5"] - d["mom20"]

    # Volatility family (14)
    for w in [5, 10, 20, 30, 60, 90, 120]:
        d[f"vol{w}"] = ret1.rolling(w).std()
    for w in [10, 20, 60, 120]:
        d[f"range_vol{w}"] = tr.rolling(w).mean() / close
    d["realized_skew20"] = ret1.rolling(20).skew()
    d["realized_kurt20"] = ret1.rolling(20).kurt()
    d["vol_of_vol20"] = d["vol20"].rolling(20).std()

    # Trend family (12)
    for w in [5, 10, 20, 50, 100, 200]:
        d[f"sma{w}"] = close.rolling(w).mean()
        d[f"ema{w}"] = close.ewm(span=w, adjust=False).mean()
    d["trend_20_50"] = _safe_div(d["sma20"], d["sma50"]) - 1
    d["trend_50_200"] = _safe_div(d["sma50"], d["sma200"]) - 1
    d["trend_ema_gap"] = _safe_div(d["ema20"], d["ema50"]) - 1
    d["breakout_60"] = _safe_div(close, high.rolling(60).max()) - 1
    d["breakdown_60"] = _safe_div(close, low.rolling(60).min()) - 1

    # Carry / term-structure proxies (8)
    d["overnight_gap"] = _safe_div(open_, close.shift(1)) - 1
    d["intraday_ret"] = _safe_div(close, open_) - 1
    d["hl_spread"] = _safe_div(high - low, close)
    d["co_spread"] = _safe_div(close - open_, close)
    d["basis_proxy_5"] = d["overnight_gap"].rolling(5).mean()
    d["basis_proxy_20"] = d["overnight_gap"].rolling(20).mean()
    d["carry_proxy_20"] = d["mom20"] - d["vol20"]
    d["carry_proxy_60"] = d["mom60"] - d["vol60"]

    # Seasonality / calendar (6)
    d["month"] = d["date"].dt.month
    d["dayofweek"] = d["date"].dt.dayofweek
    d["weekofyear"] = d["date"].dt.isocalendar().week.astype(float)
    d["seasonality_month_mean"] = ret1.groupby(d["month"]).transform("mean")
    d["seasonality_dow_mean"] = ret1.groupby(d["dayofweek"]).transform("mean")
    d["turn_of_month"] = ((d["date"].dt.day <= 3) | (d["date"].dt.day >= 27)).astype(float)

    # Oscillator family (9)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs14 = _safe_div(up.rolling(14).mean(), down.rolling(14).mean())
    rs28 = _safe_div(up.rolling(28).mean(), down.rolling(28).mean())
    d["rsi14"] = 100 - (100 / (1 + rs14))
    d["rsi28"] = 100 - (100 / (1 + rs28))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    d["stoch_k"] = 100 * _safe_div(close - low14, high14 - low14)
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()
    d["atr14"] = tr.rolling(14).mean()
    d["atr_norm"] = _safe_div(d["atr14"], close)

    # Microstructure / liquidity proxies (7)
    d["amihud_20"] = _safe_div(ret1.abs(), volm).rolling(20).mean()
    d["vol_chg_5"] = volm.pct_change(5)
    d["vol_zscore_20"] = (volm - volm.rolling(20).mean()) / volm.rolling(20).std()
    d["close_to_vwap_proxy"] = _safe_div(close, (high + low + close) / 3) - 1
    d["up_down_ratio_20"] = _safe_div((ret1 > 0).rolling(20).sum(), (ret1 < 0).rolling(20).sum())
    d["ret_autocorr_20"] = ret1.rolling(20).corr(ret1.shift(1))
    d["ret_autocorr_60"] = ret1.rolling(60).corr(ret1.shift(1))

    # Higher moments / tail risk (8)
    for w in [10, 20, 60]:
        d[f"downside_vol{w}"] = ret1.clip(upper=0).rolling(w).std()
        d[f"upside_vol{w}"] = ret1.clip(lower=0).rolling(w).std()
    d["tail_ratio_20"] = _safe_div(d["upside_vol20"], d["downside_vol20"])
    d["max_dd_60"] = _safe_div(close, close.rolling(60).max()) - 1
    d["max_uu_60"] = _safe_div(close, close.rolling(60).min()) - 1

    # Relative strength / ranking (8)
    for w in [10, 20, 60, 120]:
        d[f"zret{w}"] = (ret1 - ret1.rolling(w).mean()) / ret1.rolling(w).std()
    d["price_zscore_20"] = (close - close.rolling(20).mean()) / close.rolling(20).std()
    d["price_zscore_60"] = (close - close.rolling(60).mean()) / close.rolling(60).std()
    d["range_ratio_20_60"] = _safe_div((high - low).rolling(20).mean(), (high - low).rolling(60).mean())
    d["volume_ratio_20_60"] = _safe_div(volm.rolling(20).mean(), volm.rolling(60).mean())

    # Regime and stability (6)
    d["trend_stability_20"] = _safe_div(d["mom20"], d["vol20"])
    d["trend_stability_60"] = _safe_div(d["mom60"], d["vol60"])
    d["vol_regime"] = d["vol20"] > d["vol60"]
    d["momentum_regime"] = d["mom20"] > 0
    d["regime_score"] = d[["vol_regime", "momentum_regime"]].astype(float).mean(axis=1)
    d["close_open_gap_std_20"] = d["overnight_gap"].rolling(20).std()

    d = d.replace([np.inf, -np.inf], np.nan)
    return d
