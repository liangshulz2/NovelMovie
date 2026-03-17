"""Chronos CTA V3 main entrypoint."""

from __future__ import annotations

import pandas as pd

from config import (
    CAPITAL,
    DEVICE,
    LOOKBACK,
    MAX_PORTFOLIO_RISK,
    MAX_POSITION,
    MIN_HISTORY,
    MODEL_PATH,
    PRED_LEN,
    SYMBOLS,
    TARGET_VOL,
)
from data.futures_loader import load_futures
from factors.factor_library import compute_factors
from factors.factor_selector import select_features
from model.chronos_model import ChronosModel
from model.ensemble_model import bayesian_model_averaging
from model.xgb_model import XGBModel
from alpha.alpha_engine import generate_signal
from portfolio.portfolio_engine import final_position, kelly_scale, volatility_target
from portfolio.risk_parity import apply_risk_budget
from portfolio.optimizer import correlation_matrix, optimize_portfolio
from risk.risk_engine import portfolio_risk


def run() -> pd.DataFrame:
    chronos = ChronosModel(MODEL_PATH, DEVICE)
    xgb = XGBModel()

    records = []
    alpha_series = {}
    return_series = {}

    for symbol in SYMBOLS:
        df = compute_factors(load_futures(symbol)).tail(LOOKBACK).copy()
        if len(df) < MIN_HISTORY:
            continue

        df["target"] = df["close"].pct_change().shift(-1)
        feats = select_features(df.dropna(), "target", top_n=30)
        model_df = df[[*feats, "target"]].dropna()
        if len(model_df) < MIN_HISTORY:
            continue

        x_train = model_df[feats].iloc[:-1]
        y_train = model_df["target"].iloc[:-1]
        x_test = model_df[feats].iloc[[-1]]

        xgb.fit(x_train, y_train)
        ml_pred = float(xgb.predict(x_test)[0])

        close = df["close"].dropna()
        factor_mat = df[feats].fillna(0)
        chronos_pred = chronos.predict(close, factor_mat, PRED_LEN)
        
        # ========== 修复后的 chronos_alpha 计算 ==========
        # 健壮处理，兼容标量、数组、numpy数组、张量等
        # ========== 修复后的 chronos_alpha 计算 ==========
        # 健壮处理，兼容标量、数组、numpy数组、张量等
        if hasattr(chronos_pred, '__len__') and len(chronos_pred) > 0:
            # 提取base_value并确保是标量（处理嵌套数组/张量/多维数组）
            base_value = chronos_pred[0]
            
            # 步骤1：处理numpy/torch张量/数组（兼容多维）
            if hasattr(base_value, 'item'):
                # 先展平数组/张量，再取第一个元素（避免多维/多元素问题）
                if base_value.size > 1:
                    base_value = base_value.flatten()[0]  # 展平后取第一个元素
                base_value = base_value.item()  # 转Python标量
            # 步骤2：处理列表/元组（兼容嵌套）
            elif isinstance(base_value, (list, tuple)):
                # 递归展平嵌套结构，取第一个非空元素
                def flatten(x):
                    for item in x:
                        if isinstance(item, (list, tuple)):
                            yield from flatten(item)
                        else:
                            yield item
                flat_list = list(flatten(base_value))
                base_value = flat_list[0] if flat_list else 0.0
            # 步骤3：确保base_value是数值类型
            base_value = float(base_value) if isinstance(base_value, (int, float)) else 0.0
            
            # 计算mean_value并确保是标量
            if hasattr(chronos_pred, 'mean'):
                mean_value = chronos_pred.mean()
            else:
                mean_value = sum(chronos_pred) / len(chronos_pred)
            # 处理mean_value的数组/张量类型
            if hasattr(mean_value, 'item'):
                mean_value = mean_value.item()
            mean_value = float(mean_value) if isinstance(mean_value, (int, float)) else 0.0
            
            # 除零保护 + 转换为标量
            if abs(base_value) < 1e-8:  # 避免除以0
                chronos_alpha = 0.0
            else:
                chronos_alpha = float((mean_value - base_value) / base_value)
        elif isinstance(chronos_pred, (float, int)):
            chronos_alpha = 0.0  # 若业务需要用pred值，可改为 chronos_alpha = float(chronos_pred)
        else:
            chronos_alpha = 0.0
        # ========== 修复结束 ==========
        # ========== 修复结束 ==========

        alpha, model_weights = bayesian_model_averaging(
            predictions={"chronos": chronos_alpha, "xgb": ml_pred},
            errors={
                "chronos": abs(float(model_df["target"].iloc[-1] - chronos_alpha)) + 1e-6,
                "xgb": abs(float(model_df["target"].iloc[-1] - ml_pred)) + 1e-6,
            },
        )
        signal = generate_signal(alpha)

        vol = float(df["vol20"].iloc[-1]) if "vol20" in df.columns else float(df["close"].pct_change().rolling(20).std().iloc[-1])
        base = volatility_target(alpha, vol, TARGET_VOL)
        kelly = kelly_scale(alpha, vol)
        position = final_position(signal, base, kelly, MAX_POSITION)

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
            }
        )
        alpha_series[symbol] = df["target"].tail(60).reset_index(drop=True)
        return_series[symbol] = df["target"].tail(60).reset_index(drop=True)

    # 后续代码不变...

    out = pd.DataFrame(records)
    if out.empty:
        return out

    out["position"] = apply_risk_budget(
        out.set_index("symbol")["raw_position"],
        out.set_index("symbol")["vol20"],
        MAX_PORTFOLIO_RISK,
    ).values

    ret_df = pd.DataFrame(return_series).dropna(how="all")
    if not ret_df.empty and ret_df.shape[1] > 1:
        corr = correlation_matrix(ret_df)
        rp_w = optimize_portfolio(ret_df.fillna(0), method="risk_parity")
        out["risk_parity_weight"] = out["symbol"].map(rp_w).fillna(0.0)
        out["corr_avg"] = out["symbol"].map(corr.mean()).fillna(0.0)
        out["position"] = out["position"] * (0.5 + out["risk_parity_weight"])

    out["notional"] = out["position"] * CAPITAL
    out["portfolio_risk"] = portfolio_risk(
        out.set_index("symbol")["position"], out.set_index("symbol")["vol20"]
    )
    return out


if __name__ == "__main__":
    result = run()
    if result.empty:
        print("No tradable symbols generated for the current sample.")
    else:
        print(result[["symbol", "alpha", "signal", "position", "notional"]].to_string(index=False))
