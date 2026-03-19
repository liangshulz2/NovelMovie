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
    hold = d.get("hold", pd.Series(np.nan, index=d.index)).astype(float)
    near_close = d.get("near_close", close).astype(float)
    next_close = d.get("next_close", pd.Series(np.nan, index=d.index)).astype(float)
    far_close = d.get("far_close", pd.Series(np.nan, index=d.index)).astype(float)
    days_to_expiry = d.get("days_to_expiry", pd.Series(np.nan, index=d.index)).astype(float)

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
    features["rv_short_long_ratio"] = _safe_div(features["vol5"], features["vol20"])
    features["atr_breakout_zscore"] = (
        (features["atr14"] - features["atr14"].rolling(20).mean()) / features["atr14"].rolling(20).std()
    )
    jump_part = (open_ - close.shift(1)).abs()
    range_part = (high - low).replace(0, np.nan)
    features["vol_jump_ratio"] = _safe_div(jump_part, range_part).rolling(20).mean()

    # Futures term structure / roll factors (degrades safely to NaN when curve data is unavailable)
    annualizer = 365 / days_to_expiry.clip(lower=1)
    spread_near_next = _safe_div(near_close - next_close, near_close)
    spread_next_far = _safe_div(next_close - far_close, next_close)
    features["roll_yield_annualized"] = spread_near_next * annualizer
    features["curve_slope"] = spread_near_next
    features["curve_curvature"] = spread_near_next - spread_next_far
    features["carry_stability_20"] = _safe_div(
        features["roll_yield_annualized"].rolling(20).mean(),
        features["roll_yield_annualized"].rolling(20).std(),
    )
    spot_proxy = d.get("spot_proxy", pd.Series(np.nan, index=d.index)).astype(float)
    features["basis_to_spot_proxy"] = _safe_div(near_close, spot_proxy) - 1

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
    features["price_up_oi_up"] = ((ret1 > 0) & (hold_chg_1 > 0)).astype(float)
    features["price_up_oi_down"] = ((ret1 > 0) & (hold_chg_1 < 0)).astype(float)
    features["price_down_oi_up"] = ((ret1 < 0) & (hold_chg_1 > 0)).astype(float)
    oi_state = np.select(
        [
            (ret1 > 0) & (hold_chg_1 > 0),
            (ret1 > 0) & (hold_chg_1 < 0),
            (ret1 < 0) & (hold_chg_1 > 0),
            (ret1 < 0) & (hold_chg_1 < 0),
        ],
        [1.0, 2.0, 3.0, 4.0],
        default=np.nan,
    )
    oi_state_series = pd.Series(oi_state, index=d.index, dtype=float)
    state_switch = oi_state_series.ne(oi_state_series.shift(1)).fillna(True).cumsum()
    features["oi_regime_persist_10"] = oi_state_series.groupby(state_switch).cumcount() + 1

    # Contract-roll / liquidity-shock controls
    features["days_to_roll"] = days_to_expiry
    features["roll_window_dummy"] = days_to_expiry.le(5).astype(float)
    hold_mean_20 = hold.rolling(20).mean()
    hold_std_20 = hold.rolling(20).std()
    features["liquidity_shock"] = (
        (features["vol_zscore_20"].fillna(0.0))
        + ((hold - hold_mean_20) / hold_std_20.replace(0, np.nan)).fillna(0.0)
    )
    features["impact_proxy"] = _safe_div(features["hl_spread"], volm).rolling(20).mean()

    factor_df = pd.DataFrame(features, index=d.index)
    out = pd.concat([d, factor_df], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
