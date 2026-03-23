"""Chronos CTA V3 main entrypoint."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chronos_cta_v3.config import (
    CAPITAL,
    DEVICE,
    LOOKBACK,
    MARKET,
    MAX_PORTFOLIO_RISK,
    MAX_POSITION,
    MIN_HISTORY,
    MODEL_PATH,
    PRED_LEN,
    SEED,
    STOCK_CSV_DIR,
    STOCK_LIST_FILE,
    SYMBOLS,
    TARGET_VOL,
    runtime_config_summary,
)
from chronos_cta_v3.data.futures_loader import load_futures
from chronos_cta_v3.data.stock_loader import load_stock_from_csv, load_stock_list
from chronos_cta_v3.factors.factor_library import compute_factors
from chronos_cta_v3.factors.factor_selector import select_features
from chronos_cta_v3.model.chronos_model import ChronosModel
from chronos_cta_v3.model.ensemble_model import bayesian_model_averaging
from chronos_cta_v3.model.xgb_model import XGBModel
from chronos_cta_v3.alpha.alpha_engine import generate_signal
from chronos_cta_v3.portfolio.portfolio_engine import final_position, kelly_scale, volatility_target
from chronos_cta_v3.portfolio.risk_parity import apply_risk_budget
from chronos_cta_v3.portfolio.optimizer import correlation_matrix, optimize_portfolio
from chronos_cta_v3.risk.risk_engine import portfolio_risk


def _to_1d_array(pred) -> np.ndarray:
    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()
    elif hasattr(pred, "cpu") and hasattr(pred, "numpy"):
        pred = pred.cpu().numpy()
    arr = np.asarray(pred, dtype=float)
    return np.ravel(arr)


def _compute_chronos_alpha(pred) -> float:
    arr = _to_1d_array(pred)
    if arr.size == 0:
        return 0.0
    first = float(arr[0])
    if arr.size == 1:
        return first
    denom = abs(first)
    if denom < 1e-8:
        return 0.0
    return float((arr.mean() - first) / denom)


def _compute_oos_mae(
    model_df: pd.DataFrame,
    feats: list[str],
    chronos: ChronosModel,
    xgb: XGBModel,
    oos_window: int = 20,
    eval_step: int = 3,
) -> tuple[float, float]:
    if len(model_df) < max(MIN_HISTORY, oos_window + 2):
        target_last = float(model_df["target"].iloc[-1])
        return abs(target_last) + 1e-6, abs(target_last) + 1e-6

    chronos_errors: list[float] = []
    xgb_errors: list[float] = []
    start_idx = len(model_df) - oos_window - 1

    for i in range(start_idx, len(model_df) - 1, max(1, eval_step)):
        train = model_df.iloc[: i + 1]
        true_y = float(model_df["target"].iloc[i + 1])

        xgb.fit(train[feats], train["target"])
        xgb_pred = float(xgb.predict(model_df[feats].iloc[[i + 1]])[0])
        xgb_errors.append(abs(true_y - xgb_pred))

        close_hist = train["close"] if "close" in train.columns else train.index.to_series().astype(float)
        factor_hist = train[feats].fillna(0)
        chronos_pred = chronos.predict(close_hist, factor_hist, pred_len=PRED_LEN)
        chronos_alpha = _compute_chronos_alpha(chronos_pred)
        chronos_errors.append(abs(true_y - chronos_alpha))

    return float(np.mean(chronos_errors) + 1e-6), float(np.mean(xgb_errors) + 1e-6)


def run() -> pd.DataFrame:
    np.random.seed(SEED)
    try:
        import torch

        torch.manual_seed(SEED)
    except Exception:
        pass
    print(f"[Chronos CTA V3] Runtime config: {runtime_config_summary()}")
    try:
        chronos = ChronosModel(MODEL_PATH, DEVICE)
    except Exception as exc:
        raise RuntimeError(
            "ChronosModel initialization failed. Please check CHRONOS_MODEL_PATH and CHRONOS_DEVICE. "
            f"Current values: MODEL_PATH={MODEL_PATH}, DEVICE={DEVICE}."
        ) from exc
    xgb = XGBModel(random_state=SEED)

    records = []
    returns_history = {}

    symbols = SYMBOLS
    if MARKET not in {"futures", "stocks"}:
        raise ValueError(f"Unsupported CHRONOS_MARKET={MARKET}. Use 'futures' or 'stocks'.")
    if MARKET == "stocks" and not symbols:
        symbols = load_stock_list(STOCK_LIST_FILE)

    for symbol in symbols:
        if MARKET == "stocks":
            raw_df = load_stock_from_csv(symbol, STOCK_CSV_DIR)
        else:
            raw_df = load_futures(symbol)
        df = compute_factors(raw_df).tail(LOOKBACK).copy()
        if len(df) < MIN_HISTORY:
            continue

        df["target"] = df["close"].pct_change().shift(-1)
        feats = select_features(df.dropna(), "target", top_n=30)
        if not feats:
            fallback_feats = [c for c in ["mom5", "vol20", "atr14"] if c in df.columns]
            if not fallback_feats:
                continue
            feats = fallback_feats
        model_df = df[["close", *feats, "target"]].dropna()
        if len(model_df) < MIN_HISTORY:
            continue

        x_train = model_df[feats].iloc[:-1]
        y_train = model_df["target"].iloc[:-1]
        x_test = model_df[feats].iloc[[-1]]

        xgb.fit(x_train, y_train)
        ml_pred = float(xgb.predict(x_test)[0])

        close = model_df["close"].dropna()
        factor_mat = model_df[feats].fillna(0)
        chronos_pred = chronos.predict(close, factor_mat, PRED_LEN)
        chronos_alpha = _compute_chronos_alpha(chronos_pred)

        chronos_mae, xgb_mae = _compute_oos_mae(model_df, feats, chronos, xgb)
        alpha, model_weights = bayesian_model_averaging(
            predictions={"chronos": chronos_alpha, "xgb": ml_pred},
            errors={"chronos": chronos_mae, "xgb": xgb_mae},
        )
        signal = generate_signal(alpha)

        latest_close = float(df["close"].iloc[-1])
        vol = float(df["vol20"].iloc[-1]) if "vol20" in df.columns else float(df["close"].pct_change().rolling(20).std().iloc[-1])
        atr14 = float(df["atr14"].iloc[-1]) if "atr14" in df.columns and pd.notna(df["atr14"].iloc[-1]) else latest_close * max(vol, 0.005)
        base = volatility_target(vol, TARGET_VOL)
        kelly = kelly_scale(alpha, vol)
        position = final_position(signal, base, kelly, MAX_POSITION)

        stop_loss_price = latest_close - signal * 1.5 * atr14
        target_price = latest_close + signal * 3.0 * atr14

        records.append(
            {
                "symbol": symbol,
                "alpha": alpha,
                "ml_pred": ml_pred,
                "signal": signal,
                "vol20": vol,
                "raw_position": position,
                "weight_chronos": model_weights.get("chronos", 0.0),
                "weight_xgb": model_weights.get("xgb", 0.0),
                "stop_loss_price": stop_loss_price,
                "target_price": target_price,
            }
        )
        returns_history[symbol] = model_df["target"].tail(60).reset_index(drop=True)

    out = pd.DataFrame(records)
    if out.empty:
        return out

    out["position"] = apply_risk_budget(
        out.set_index("symbol")["raw_position"],
        out.set_index("symbol")["vol20"],
        MAX_PORTFOLIO_RISK,
    ).values

    ret_df = pd.DataFrame(returns_history).dropna(how="all")
    cov_df = pd.DataFrame()
    if not ret_df.empty and ret_df.shape[1] > 1:
        corr = correlation_matrix(ret_df)
        cov_df = ret_df.cov()
        rp_w = optimize_portfolio(ret_df.fillna(0), method="risk_parity")
        out["risk_parity_weight"] = out["symbol"].map(rp_w).fillna(0.0)
        out["corr_avg"] = out["symbol"].map(corr.mean()).fillna(0.0)
        out["position"] = out["position"] * (0.5 + out["risk_parity_weight"])
        out["position"] = apply_risk_budget(
            out.set_index("symbol")["position"],
            out.set_index("symbol")["vol20"],
            MAX_PORTFOLIO_RISK,
            cov_matrix=cov_df,
        ).values
    out["position"] = out["position"].clip(lower=-MAX_POSITION, upper=MAX_POSITION)
    exceed_count = int((out["position"].abs() > (MAX_POSITION + 1e-12)).sum())
    print(
        f"[Chronos CTA V3] Post-risk position clip check: exceed_count={exceed_count}, "
        f"max_abs_position={out['position'].abs().max():.6f}"
    )
    assert (out["position"].abs() <= (MAX_POSITION + 1e-12)).all(), "Position exceeds MAX_POSITION after final clip."

    out["notional"] = out["position"] * CAPITAL
    using_cov = not cov_df.empty
    out["portfolio_risk"] = portfolio_risk(
        out.set_index("symbol")["position"],
        out.set_index("symbol")["vol20"],
        cov_matrix=cov_df if using_cov else None,
    )
    out["portfolio_risk_method"] = "covariance" if using_cov else "diag_vol"
    return out


if __name__ == "__main__":
    result = run()
    if result.empty:
        print("No tradable symbols generated for the current sample.")
    else:
        print(
            result[
                [
                    "symbol",
                    "alpha",
                    "signal",
                    "position",
                    "notional",
                    "stop_loss_price",
                    "target_price",
                ]
            ].to_string(index=False)
        )
