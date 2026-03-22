"""Factor selection and IC analysis utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError


_EXCLUDE_BASE = {"date", "open", "high", "low", "close"}


def candidate_numeric_features(df: pd.DataFrame, target_col: str) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in _EXCLUDE_BASE | {target_col}
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def factor_ic_analysis(df: pd.DataFrame, target_col: str, method: str = "spearman") -> pd.DataFrame:
    """Automatically calculate IC metrics for all candidate factors."""
    feature_cols = candidate_numeric_features(df, target_col)
    rows = []
    for col in feature_cols:
        pair = df[[col, target_col]].dropna()
        if len(pair) < 20:
            continue
        ic = pair[col].corr(pair[target_col], method=method)
        rows.append(
            {
                "factor": col,
                "ic": float(ic) if pd.notna(ic) else np.nan,
                "abs_ic": abs(float(ic)) if pd.notna(ic) else np.nan,
                "sample_size": int(len(pair)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["ic_rank"] = out["abs_ic"].rank(ascending=False, method="dense")
    return out.sort_values(["abs_ic", "sample_size"], ascending=[False, False]).reset_index(drop=True)


def select_features(df: pd.DataFrame, target_col: str, top_n: int = 30) -> list[str]:
    """Select features with strongest absolute IC as default behavior."""
    analysis = factor_ic_analysis(df, target_col)
    if analysis.empty:
        return []
    return analysis.head(top_n)["factor"].tolist()


def _safe_vif(df: pd.DataFrame, feature: str, others: list[str]) -> float:
    """Compute VIF(feature | others) with a small ridge fallback for stability."""
    if not others:
        return 1.0

    work = df[[feature] + others].dropna()
    if len(work) < max(20, len(others) + 5):
        return np.nan

    y = work[feature].to_numpy(dtype=float)
    x = work[others].to_numpy(dtype=float)

    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std[x_std == 0] = 1.0
    x = (x - x_mean) / x_std

    y_mean = y.mean()
    y_std = y.std()
    if y_std == 0:
        return np.nan
    y = (y - y_mean) / y_std

    # Add intercept.
    x_design = np.column_stack([np.ones(len(x)), x])

    try:
        beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    except LinAlgError:
        return np.nan

    pred = x_design @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return np.nan

    r2 = 1.0 - ss_res / ss_tot
    # Numerical guard.
    r2 = min(max(r2, 0.0), 0.999999)
    return 1.0 / (1.0 - r2)


def secondary_feature_filter(
    df: pd.DataFrame,
    target_col: str,
    ranked_features: list[str],
    corr_threshold: float = 0.85,
    vif_threshold: float = 10.0,
    max_features: int | None = None,
) -> list[str]:
    """Second-stage redundancy filter using pairwise correlation + VIF.

    Args:
        df: input frame with candidate features.
        target_col: label column (excluded from redundancy checks).
        ranked_features: first-stage ranking list (priority from high to low).
        corr_threshold: absolute pairwise correlation cutoff.
        vif_threshold: VIF cutoff for multi-collinearity control.
        max_features: optional cap for selected feature count.

    Returns:
        Feature list after redundancy filtering while preserving ranking priority.
    """
    usable = [
        c
        for c in ranked_features
        if c in df.columns and c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not usable:
        return []

    selected: list[str] = []
    for col in usable:
        if max_features is not None and len(selected) >= max_features:
            break

        # Step1: pairwise correlation screen with already accepted factors.
        too_close = False
        for s in selected:
            pair = df[[col, s]].dropna()
            if len(pair) < 20:
                continue
            corr = pair[col].corr(pair[s], method="spearman")
            if pd.notna(corr) and abs(float(corr)) >= corr_threshold:
                too_close = True
                break
        if too_close:
            continue

        # Step2: VIF screen once there are enough selected factors.
        if selected:
            vif = _safe_vif(df, col, selected)
            if pd.notna(vif) and vif >= vif_threshold:
                continue

        selected.append(col)

    return selected


def select_features_two_stage(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 30,
    corr_threshold: float = 0.85,
    vif_threshold: float = 10.0,
    max_features: int | None = None,
) -> list[str]:
    """Two-stage selector: IC ranking first, then redundancy filter."""
    stage1 = select_features(df, target_col=target_col, top_n=top_n)
    if not stage1:
        return []
    return secondary_feature_filter(
        df=df,
        target_col=target_col,
        ranked_features=stage1,
        corr_threshold=corr_threshold,
        vif_threshold=vif_threshold,
        max_features=max_features,
    )
