"""Factor selection utilities."""

import pandas as pd


def select_features(df: pd.DataFrame, target_col: str, top_n: int = 30) -> list[str]:
    candidate_cols = [
        c
        for c in df.columns
        if c not in {"date", "open", "high", "low", "close", target_col}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    corr = (
        df[candidate_cols + [target_col]]
        .corr(numeric_only=True)[target_col]
        .drop(labels=[target_col])
        .abs()
        .sort_values(ascending=False)
    )
    return corr.head(top_n).index.tolist()
