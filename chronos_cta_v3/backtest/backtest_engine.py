"""Backtesting utilities and walk-forward platform."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADE_COLUMNS = [
    "date",
    "action",
    "from_position",
    "to_position",
    "prediction",
    "price",
    "exec_price",
    "price_slippage",
    "transaction_cost",
    "pnl",
    "equity",
]


def _validate_series_index(name: str, series: pd.Series) -> None:
    if not isinstance(series.index, pd.DatetimeIndex):
        return
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonic increasing.")
    if series.index.has_duplicates:
        raise ValueError(f"{name} index contains duplicated timestamps.")


def backtest(
    returns: pd.Series,
    positions: pd.Series,
    commission: float = 0.0,
    slippage: float = 0.0,
) -> pd.Series:
    """Simple backtest with optional transaction costs."""
    aligned_returns = returns.fillna(0).astype(float)
    aligned_pos = positions.reindex(aligned_returns.index).fillna(0).astype(float)

    prev_pos = aligned_pos.shift(1).fillna(0)
    turnover = (aligned_pos - prev_pos).abs()
    cost_rate = float(max(commission, 0.0) + max(slippage, 0.0))

    pnl = aligned_returns * prev_pos - turnover * cost_rate
    equity = (1 + pnl).cumprod()
    return equity


def auto_generate_strategy(predictions: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Auto strategy generation from alpha predictions."""
    pos = pd.Series(0, index=predictions.index, dtype=float)
    pos[predictions > threshold] = 1.0
    pos[predictions < -threshold] = -1.0
    return pos


def classify_regime(
    predictions: pd.Series,
    returns: pd.Series,
    vol_lookback: int = 20,
    trend_lookback: int = 20,
) -> pd.Series:
    """Three-state regime classification: trend/range/event."""
    aligned_ret = returns.reindex(predictions.index).fillna(0.0)
    short_vol = aligned_ret.rolling(vol_lookback).std()
    long_vol = aligned_ret.rolling(60).std()
    vol_z = (short_vol - short_vol.rolling(60).mean()) / short_vol.rolling(60).std()
    trend_stability = predictions.rolling(trend_lookback).mean() / short_vol.replace(0, np.nan)

    regime = pd.Series("range", index=predictions.index, dtype=object)
    regime[vol_z > 1.25] = "event"
    regime[(regime != "event") & (trend_stability.abs() > 0.5)] = "trend"
    return regime


def apply_execution_controls(
    target_pos: pd.Series,
    prev_position: float = 0.0,
    no_trade_band: float = 0.0,
    position_smoothing: float = 1.0,
) -> pd.Series:
    """Apply no-trade band and EWMA-like position smoothing."""
    out = pd.Series(index=target_pos.index, dtype=float)
    last_pos = float(prev_position)
    smooth = float(np.clip(position_smoothing, 0.0, 1.0))
    band = float(max(no_trade_band, 0.0))

    for idx, desired in target_pos.items():
        desired = float(desired)
        raw_new = smooth * desired + (1.0 - smooth) * last_pos
        if abs(raw_new - last_pos) < band:
            new_pos = last_pos
        else:
            new_pos = raw_new
        out.loc[idx] = new_pos
        last_pos = new_pos
    return out


def optimize_signal_threshold(predictions: pd.Series, returns: pd.Series, grid: list[float] | None = None) -> float:
    """Automatic parameter optimization via simple grid-search on Sharpe."""
    if grid is None:
        grid = [0.0, 0.0025, 0.005, 0.01, 0.02]

    best_thr = grid[0]
    best_score = -np.inf
    for thr in grid:
        pos = auto_generate_strategy(predictions, threshold=thr)
        pnl = returns.fillna(0) * pos.shift(1).fillna(0)
        std = pnl.std()
        score = 0 if std == 0 else np.sqrt(252) * pnl.mean() / std
        if score > best_score:
            best_score = score
            best_thr = thr
    return float(best_thr)


def walk_forward_backtest(
    predictions: pd.Series,
    returns: pd.Series,
    train_window: int = 120,
    test_window: int = 20,
    start_date: str | pd.Timestamp | None = None,
    commission: float = 0.0,
    slippage: float = 0.0,
    no_trade_band: float = 0.0,
    position_smoothing: float = 1.0,
    regime_event_position_scale: float = 0.5,
) -> pd.DataFrame:
    """Walk-forward backtest with threshold re-fit per window and cost model."""
    _validate_series_index("predictions", predictions)
    _validate_series_index("returns", returns)
    if isinstance(predictions.index, pd.DatetimeIndex) and isinstance(returns.index, pd.DatetimeIndex):
        if str(predictions.index.tz) != str(returns.index.tz):
            raise ValueError("predictions and returns index timezones are inconsistent.")

    returns = returns.reindex(predictions.index).fillna(0.0)
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        mask = predictions.index >= start_ts
        predictions = predictions.loc[mask]
        returns = returns.reindex(predictions.index).fillna(0.0)

    rows = []
    n = len(predictions)
    cost_rate = float(max(commission, 0.0) + max(slippage, 0.0))
    prev_position = 0.0

    for start in range(train_window, n, test_window):
        train_slice = slice(start - train_window, start)
        test_slice = slice(start, min(start + test_window, n))

        thr = optimize_signal_threshold(predictions.iloc[train_slice], returns.iloc[train_slice])
        test_pred = predictions.iloc[test_slice]
        test_ret = returns.iloc[test_slice]
        target_pos = auto_generate_strategy(test_pred, threshold=thr)
        regime = classify_regime(test_pred, test_ret)
        target_pos = apply_execution_controls(
            target_pos=target_pos,
            prev_position=prev_position,
            no_trade_band=no_trade_band,
            position_smoothing=position_smoothing,
        )
        target_pos = target_pos.where(regime != "event", target_pos * float(np.clip(regime_event_position_scale, 0.0, 1.0)))

        for i in range(len(test_pred)):
            desired_pos = float(target_pos.iloc[i])
            turnover = abs(desired_pos - prev_position)
            trading_cost = turnover * cost_rate
            realized_pnl = float(test_ret.iloc[i]) * prev_position - trading_cost

            rows.append(
                {
                    "date": test_pred.index[i],
                    "prediction": float(test_pred.iloc[i]),
                    "position": desired_pos,
                    "prev_position": prev_position,
                    "return": float(test_ret.iloc[i]),
                    "turnover": turnover,
                    "cost": trading_cost,
                    "pnl": realized_pnl,
                    "threshold": float(thr),
                    "regime": regime.iloc[i],
                }
            )
            prev_position = desired_pos

    out = pd.DataFrame(rows)
    if not out.empty:
        out["equity"] = (1 + out["pnl"]).cumprod()
    return out


def trade_records(
    daily: pd.DataFrame,
    prices: pd.Series | None = None,
    commission: float = 0.0,
    slippage: float = 0.0,
) -> pd.DataFrame:
    """Generate trade records from a backtest daily dataframe."""
    if daily.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    records = []
    prev_pos = 0.0
    price_series = prices.reindex(daily["date"]).astype(float) if prices is not None else None
    cost_rate = float(max(commission, 0.0) + max(slippage, 0.0))

    for i, row in daily.iterrows():
        new_pos = float(row["position"])
        if np.isclose(new_pos, prev_pos):
            continue

        if np.isclose(prev_pos, 0.0) and not np.isclose(new_pos, 0.0):
            action = "OPEN"
        elif not np.isclose(prev_pos, 0.0) and np.isclose(new_pos, 0.0):
            action = "CLOSE"
        elif np.sign(prev_pos) != np.sign(new_pos):
            action = "REVERSE"
        else:
            action = "ADJUST"

        raw_price = float(price_series.iloc[i]) if price_series is not None and pd.notna(price_series.iloc[i]) else np.nan
        price_slippage = abs(raw_price) * float(max(slippage, 0.0)) if pd.notna(raw_price) else np.nan
        exec_price = raw_price + np.sign(new_pos - prev_pos) * price_slippage if pd.notna(raw_price) else np.nan
        turnover = abs(new_pos - prev_pos)
        transaction_cost = turnover * cost_rate

        records.append(
            {
                "date": row["date"],
                "action": action,
                "from_position": prev_pos,
                "to_position": new_pos,
                "prediction": float(row["prediction"]),
                "price": raw_price,
                "exec_price": exec_price,
                "price_slippage": price_slippage,
                "transaction_cost": transaction_cost,
                "pnl": float(row["pnl"]),
                "equity": float(row.get("equity", np.nan)),
            }
        )
        prev_pos = new_pos

    return pd.DataFrame(records, columns=TRADE_COLUMNS)


@dataclass
class BacktestPlatform:
    """Complete backtest platform wrapper."""

    train_window: int = 120
    test_window: int = 20
    commission: float = 0.0
    slippage: float = 0.0
    no_trade_band: float = 0.0
    position_smoothing: float = 1.0
    regime_event_position_scale: float = 0.5

    def run(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        start_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return walk_forward_backtest(
            predictions=predictions,
            returns=returns,
            train_window=self.train_window,
            test_window=self.test_window,
            start_date=start_date,
            commission=self.commission,
            slippage=self.slippage,
            no_trade_band=self.no_trade_band,
            position_smoothing=self.position_smoothing,
            regime_event_position_scale=self.regime_event_position_scale,
        )

    def run_with_trade_records(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        prices: pd.Series | None = None,
        start_date: str | pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        daily = self.run(predictions=predictions, returns=returns, start_date=start_date)
        return daily, trade_records(daily, prices=prices, commission=self.commission, slippage=self.slippage)

    def cost_sensitivity(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        multipliers: list[float] | None = None,
        start_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Run walk-forward under cost multipliers (e.g. 1x~3x)."""
        if multipliers is None:
            multipliers = [1.0, 2.0, 3.0]

        rows: list[dict] = []
        for m in multipliers:
            daily = walk_forward_backtest(
                predictions=predictions,
                returns=returns,
                train_window=self.train_window,
                test_window=self.test_window,
                start_date=start_date,
                commission=self.commission * m,
                slippage=self.slippage * m,
                no_trade_band=self.no_trade_band,
                position_smoothing=self.position_smoothing,
                regime_event_position_scale=self.regime_event_position_scale,
            )
            if daily.empty:
                rows.append({"cost_multiplier": m, "final_equity": np.nan, "sharpe": np.nan, "max_drawdown": np.nan})
                continue
            pnl = daily["pnl"].fillna(0.0)
            vol = pnl.std()
            sharpe = float(0.0 if vol <= 1e-12 else np.sqrt(252) * pnl.mean() / vol)
            max_dd = float((daily["equity"] / daily["equity"].cummax() - 1).min())
            rows.append(
                {
                    "cost_multiplier": float(m),
                    "final_equity": float(daily["equity"].iloc[-1]),
                    "sharpe": sharpe,
                    "max_drawdown": max_dd,
                }
            )
        return pd.DataFrame(rows)
