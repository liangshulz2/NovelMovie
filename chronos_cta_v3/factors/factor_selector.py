"""Factor selection and IC analysis utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
