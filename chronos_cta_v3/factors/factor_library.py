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
    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    features: dict[str, pd.Series] = {}

    # Momentum family (15)
    for w in [2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 90, 120]:
        features[f"mom{w}"] = close.pct_change(w)
    features["mom_sign20"] = np.sign(features["mom20"])
    features["mom_accel"] = features["mom10"] - features["mom30"]
    features["mom_curve"] = features["mom5"] - features["mom20"]

    # Volatility family (14)
    for w in [5, 10, 20, 30, 60, 90, 120]:
        features[f"vol{w}"] = ret1.rolling(w).std()
    for w in [10, 20, 60, 120]:
        features[f"range_vol{w}"] = tr.rolling(w).mean() / close
    features["realized_skew20"] = ret1.rolling(20).skew()
    features["realized_kurt20"] = ret1.rolling(20).kurt()
    features["vol_of_vol20"] = features["vol20"].rolling(20).std()

    # Trend family (12)
    for w in [5, 10, 20, 50, 100, 200]:
        features[f"sma{w}"] = close.rolling(w).mean()
        features[f"ema{w}"] = close.ewm(span=w, adjust=False).mean()
    features["trend_20_50"] = _safe_div(features["sma20"], features["sma50"]) - 1
    features["trend_50_200"] = _safe_div(features["sma50"], features["sma200"]) - 1
    features["trend_ema_gap"] = _safe_div(features["ema20"], features["ema50"]) - 1
    features["breakout_60"] = _safe_div(close, high.rolling(60).max()) - 1
    features["breakdown_60"] = _safe_div(close, low.rolling(60).min()) - 1

    # Carry / term-structure proxies (8)
    features["overnight_gap"] = _safe_div(open_, close.shift(1)) - 1
    features["intraday_ret"] = _safe_div(close, open_) - 1
    features["hl_spread"] = _safe_div(high - low, close)
    features["co_spread"] = _safe_div(close - open_, close)
    features["basis_proxy_5"] = features["overnight_gap"].rolling(5).mean()
    features["basis_proxy_20"] = features["overnight_gap"].rolling(20).mean()
    features["carry_proxy_20"] = features["mom20"] - features["vol20"]
    features["carry_proxy_60"] = features["mom60"] - features["vol60"]

    # Seasonality / calendar (6)
    features["month"] = d["date"].dt.month
    features["dayofweek"] = d["date"].dt.dayofweek
    features["weekofyear"] = d["date"].dt.isocalendar().week.astype(float)
    features["seasonality_month_mean"] = ret1.groupby(features["month"]).transform("mean")
    features["seasonality_dow_mean"] = ret1.groupby(features["dayofweek"]).transform("mean")
    features["turn_of_month"] = ((d["date"].dt.day <= 3) | (d["date"].dt.day >= 27)).astype(float)

    # Oscillator family (9)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs14 = _safe_div(up.rolling(14).mean(), down.rolling(14).mean())
    rs28 = _safe_div(up.rolling(28).mean(), down.rolling(28).mean())
    features["rsi14"] = 100 - (100 / (1 + rs14))
    features["rsi28"] = 100 - (100 / (1 + rs28))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    features["macd"] = ema12 - ema26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    features["stoch_k"] = 100 * _safe_div(close - low14, high14 - low14)
    features["stoch_d"] = features["stoch_k"].rolling(3).mean()
    features["atr14"] = tr.rolling(14).mean()
    features["atr_norm"] = _safe_div(features["atr14"], close)

    # Microstructure / liquidity proxies (7)
    features["amihud_20"] = _safe_div(ret1.abs(), volm).rolling(20).mean()
    features["vol_chg_5"] = volm.pct_change(5)
    features["vol_zscore_20"] = (volm - volm.rolling(20).mean()) / volm.rolling(20).std()
    features["close_to_vwap_proxy"] = _safe_div(close, (high + low + close) / 3) - 1
    features["up_down_ratio_20"] = _safe_div((ret1 > 0).rolling(20).sum(), (ret1 < 0).rolling(20).sum())
    features["ret_autocorr_20"] = ret1.rolling(20).corr(ret1.shift(1))
    features["ret_autocorr_60"] = ret1.rolling(60).corr(ret1.shift(1))

    # Higher moments / tail risk (8)
    for w in [10, 20, 60]:
        features[f"downside_vol{w}"] = ret1.clip(upper=0).rolling(w).std()
        features[f"upside_vol{w}"] = ret1.clip(lower=0).rolling(w).std()
    features["tail_ratio_20"] = _safe_div(features["upside_vol20"], features["downside_vol20"])
    features["max_dd_60"] = _safe_div(close, close.rolling(60).max()) - 1
    features["max_uu_60"] = _safe_div(close, close.rolling(60).min()) - 1

    # Relative strength / ranking (8)
    for w in [10, 20, 60, 120]:
        features[f"zret{w}"] = (ret1 - ret1.rolling(w).mean()) / ret1.rolling(w).std()
    features["price_zscore_20"] = (close - close.rolling(20).mean()) / close.rolling(20).std()
    features["price_zscore_60"] = (close - close.rolling(60).mean()) / close.rolling(60).std()
    features["range_ratio_20_60"] = _safe_div((high - low).rolling(20).mean(), (high - low).rolling(60).mean())
    features["volume_ratio_20_60"] = _safe_div(volm.rolling(20).mean(), volm.rolling(60).mean())

    # Regime and stability (6)
    features["trend_stability_20"] = _safe_div(features["mom20"], features["vol20"])
    features["trend_stability_60"] = _safe_div(features["mom60"], features["vol60"])
    features["vol_regime"] = features["vol20"] > features["vol60"]
    features["momentum_regime"] = features["mom20"] > 0
    features["regime_score"] = pd.concat(
        [features["vol_regime"].astype(float), features["momentum_regime"].astype(float)], axis=1
    ).mean(axis=1)
    features["close_open_gap_std_20"] = features["overnight_gap"].rolling(20).std()

    factor_df = pd.DataFrame(features, index=d.index)
    out = pd.concat([d, factor_df], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
