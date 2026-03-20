"""CTA factor library (80+ factors) for cross-commodity research."""

import numpy as np
import pandas as pd


ROLL_WINDOWS = [3, 5, 10, 20, 30, 60]
INDUSTRY_GROUPS = {
    # 黑色
    "RB": "black",
    "HC": "black",
    "I": "black",
    "J": "black",
    "JM": "black",
    "SM": "black",
    "SF": "black",
    # 有色
    "CU": "nonferrous",
    "AL": "nonferrous",
    "ZN": "nonferrous",
    "PB": "nonferrous",
    "NI": "nonferrous",
    "SN": "nonferrous",
    # 化工
    "TA": "chemical",
    "MA": "chemical",
    "PP": "chemical",
    "L": "chemical",
    "V": "chemical",
    "EG": "chemical",
    "EB": "chemical",
    "SA": "chemical",
    "FG": "chemical",
    # 农产品
    "A": "agri",
    "B": "agri",
    "M": "agri",
    "Y": "agri",
    "P": "agri",
    "RM": "agri",
    "OI": "agri",
    "CF": "agri",
    "SR": "agri",
    "C": "agri",
    "CS": "agri",
}


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    def _rank_last(arr: np.ndarray) -> float:
        if len(arr) == 0 or np.isnan(arr[-1]):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        return float((valid <= arr[-1]).mean())

    return s.rolling(window).apply(_rank_last, raw=True)


def _rolling_linreg_quality(log_price: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    x = np.arange(window, dtype=float)

    def _calc_stats(arr: np.ndarray) -> tuple[float, float]:
        if np.isnan(arr).any():
            return np.nan, np.nan
        y = arr.astype(float)
        x_mean = x.mean()
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        var_x = ((x - x_mean) ** 2).sum()
        if var_x <= 0:
            return np.nan, np.nan
        slope = cov / var_x
        intercept = y_mean - slope * x_mean
        fitted = intercept + slope * x
        resid = y - fitted
        sst = ((y - y_mean) ** 2).sum()
        sse = (resid**2).sum()
        r2 = 1 - sse / sst if sst > 0 else np.nan
        resid_std = resid.std(ddof=0)
        slope_over_noise = slope / resid_std if resid_std > 0 else np.nan
        return r2, slope_over_noise

    r2 = log_price.rolling(window).apply(lambda a: _calc_stats(a)[0], raw=True)
    son = log_price.rolling(window).apply(lambda a: _calc_stats(a)[1], raw=True)
    return r2, son


def _extract_product_code(symbol: str) -> str:
    if not isinstance(symbol, str):
        return ""
    return "".join(ch for ch in symbol.upper() if ch.isalpha())


def _industry_group(symbol: str) -> str:
    code = _extract_product_code(symbol)
    return INDUSTRY_GROUPS.get(code, "other")


def add_cross_sectional_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Append cross-sectional factors when date+symbol panel data is available."""
    out = df.copy()
    required_cols = {"date", "symbol"}
    if not required_cols.issubset(out.columns):
        return out

    if out["symbol"].nunique(dropna=True) <= 1:
        return out

    if "mom20" in out.columns:
        out["cs_mom_rank"] = out.groupby("date")["mom20"].rank(pct=True)
    if "mom5" in out.columns:
        out["cs_reversal_5"] = out.groupby("date")["mom5"].rank(pct=True, ascending=True)
    if "vol20" in out.columns:
        out["cs_vol_rank"] = out.groupby("date")["vol20"].rank(pct=True)
    if "oi_chg_20" in out.columns:
        out["cs_oi_chg_rank"] = out.groupby("date")["oi_chg_20"].rank(pct=True)

    if "close" in out.columns:
        ret20 = out.groupby("symbol")["close"].pct_change(20)
        ret20_z = ret20.groupby(out["symbol"]).transform(lambda s: (s - s.rolling(60).mean()) / s.rolling(60).std())
    else:
        ret20_z = pd.Series(np.nan, index=out.index)

    if "oi_chg_20" in out.columns:
        oi20 = out["oi_chg_20"]
        oi20_z = oi20.groupby(out["symbol"]).transform(lambda s: (s - s.rolling(60).mean()) / s.rolling(60).std())
    else:
        oi20_z = pd.Series(np.nan, index=out.index)

    if "volume_ratio_20_60" in out.columns:
        vol_ratio = out["volume_ratio_20_60"]
        vol_ratio_z = vol_ratio.groupby(out["symbol"]).transform(
            lambda s: (s - s.rolling(60).mean()) / s.rolling(60).std()
        )
    else:
        vol_ratio_z = pd.Series(np.nan, index=out.index)
    out["crowding_score"] = ret20_z + oi20_z + vol_ratio_z
    out["crowding_cs_rank"] = out.groupby("date")["crowding_score"].rank(pct=True)

    out["industry_group"] = out["symbol"].map(_industry_group)
    if "mom20" in out.columns:
        out["industry_mom_rank"] = out.groupby(["date", "industry_group"])["mom20"].rank(pct=True)
    return out


def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"].astype(float)
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    open_ = d["open"].astype(float)
    volm = d.get("volume", pd.Series(np.nan, index=d.index)).astype(float)
    hold = d.get("hold", pd.Series(np.nan, index=d.index)).astype(float)

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
    features["overnight_vol_20"] = features["overnight_gap"].rolling(20).std()
    features["overnight_skew_60"] = features["overnight_gap"].rolling(60).skew()
    features["overnight_kurt_60"] = features["overnight_gap"].rolling(60).kurt()
    # Positive value suggests gap and intraday move reinforce each other.
    features["gap_fill_1"] = features["overnight_gap"] * features["intraday_ret"]
    features["gap_sign_persist_20"] = (
        features["overnight_gap"].pipe(np.sign).eq(features["overnight_gap"].shift(1).pipe(np.sign)).rolling(20).mean()
    )

    # Seasonality / calendar (6)
    features["month"] = d["date"].dt.month
    features["dayofweek"] = d["date"].dt.dayofweek
    features["weekofyear"] = d["date"].dt.isocalendar().week.astype(float)
    # IMPORTANT: only use historical observations (shifted expanding mean) to avoid look-ahead bias.
    month_group = ret1.groupby(features["month"])
    month_hist_sum = month_group.cumsum() - ret1.fillna(0)
    month_hist_count = month_group.cumcount()
    features["seasonality_month_mean"] = month_hist_sum / month_hist_count.replace(0, np.nan)

    dow_group = ret1.groupby(features["dayofweek"])
    dow_hist_sum = dow_group.cumsum() - ret1.fillna(0)
    dow_hist_count = dow_group.cumcount()
    features["seasonality_dow_mean"] = dow_hist_sum / dow_hist_count.replace(0, np.nan)
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
    features["hl_efficiency_20"] = _safe_div((close - open_).abs(), (high - low)).rolling(20).mean()
    features["volume_sign_persist_20"] = np.sign(features["vol_chg_5"]).rolling(20).mean().abs()

    # Higher moments / tail risk (8)
    for w in [10, 20, 60]:
        features[f"downside_vol{w}"] = ret1.clip(upper=0).rolling(w).std()
        features[f"upside_vol{w}"] = ret1.clip(lower=0).rolling(w).std()
    features["tail_ratio_20"] = _safe_div(features["upside_vol20"], features["downside_vol20"])
    features["up_var_share_20"] = _safe_div(
        features["upside_vol20"], features["upside_vol20"] + features["downside_vol20"]
    )
    features["jump_ratio_20"] = _safe_div(features["overnight_vol_20"], features["vol20"])
    features["range_to_close_vol_20"] = _safe_div(high - low, close).rolling(20).std()
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
    features["vol_term"] = features["vol10"] - features["vol60"]
    features["vol_term_zscore_60"] = (features["vol_term"] - features["vol_term"].rolling(60).mean()) / features[
        "vol_term"
    ].rolling(60).std()

    # Open-interest structure (requires hold column from futures_loader; safely degrades to NaN if unavailable)
    hold_chg_1 = hold.pct_change()
    hold_chg_5 = hold.pct_change(5)
    hold_chg_20 = hold.pct_change(20)
    ret20 = close.pct_change(20)
    ret20_z = (ret20 - ret20.rolling(60).mean()) / ret20.rolling(60).std()
    oi20_z = (hold_chg_20 - hold_chg_20.rolling(60).mean()) / hold_chg_20.rolling(60).std()
    features["oi_chg_1"] = hold_chg_1
    features["oi_chg_5"] = hold_chg_5
    features["oi_chg_20"] = hold_chg_20
    features["price_oi_div"] = ret20_z - oi20_z
    features["oi_beta_60"] = ret1.rolling(60).corr(hold_chg_1)
    features["vol_oi_sync_20"] = features["vol_chg_5"].rolling(20).corr(hold_chg_5)
    features["price_volume_corr_20"] = ret1.rolling(20).corr(features["vol_chg_5"])
    features["signed_volume_20"] = _safe_div((np.sign(ret1) * volm).rolling(20).sum(), volm.rolling(20).sum())
    features["oi_price_corr_20"] = ret1.rolling(20).corr(hold_chg_1)

    # Trend quality and breakout persistence
    rolling_high_60 = high.rolling(60).max()
    features["breakout_persist_5"] = (close >= rolling_high_60 * 0.995).rolling(5).mean()
    log_close = np.log(close.replace(0, np.nan))
    trend_r2_20, slope_over_noise_20 = _rolling_linreg_quality(log_close, 20)
    features["trend_r2_20"] = trend_r2_20
    features["slope_over_noise_20"] = slope_over_noise_20

    # K-line structure factors
    candle_range = (high - low).replace(0, np.nan)
    features["upper_shadow_ratio"] = _safe_div(high - np.maximum(open_, close), candle_range)
    features["lower_shadow_ratio"] = _safe_div(np.minimum(open_, close) - low, candle_range)
    features["body_ratio"] = _safe_div((close - open_).abs(), candle_range)
    for col in ["upper_shadow_ratio", "lower_shadow_ratio", "body_ratio"]:
        features[f"{col}_mean20"] = features[col].rolling(20).mean()
        features[f"{col}_std20"] = features[col].rolling(20).std()

    # Percentile state-machine factors (time-series percentile rank)
    features["close_rank_20"] = _rolling_pct_rank(close, 20)
    features["vol_rank_60"] = _rolling_pct_rank(features["vol20"], 60)
    features["oi_rank_60"] = _rolling_pct_rank(hold, 60)

    factor_df = pd.DataFrame(features, index=d.index)
    out = pd.concat([d, factor_df], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    return add_cross_sectional_factors(out)
