"""Factor evaluation utilities (IC, quantile return, recommendation)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class EvalConfig:
    date_col = "date"
    symbol_col = "symbol"
    close_col = "close"

    horizon = 1
    n_quantiles = 10
    min_cs_size = 20
    min_ts_size = 60
    ic_mode = "auto"  # auto / cs / ts

    ic_mean_keep = 0.01
    ic_t_keep = 2.0
    ls_t_keep = 2.0
    mono_keep = 0.6

    ic_mean_watch = 0.005
    ic_t_watch = 1.0
    ls_t_watch = 1.0
    mono_watch = 0.3


def add_forward_return(df: pd.DataFrame, cfg: EvalConfig) -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values([cfg.symbol_col, cfg.date_col]).reset_index(drop=True)
    d[f"fwd_ret_{cfg.horizon}"] = (
        d.groupby(cfg.symbol_col)[cfg.close_col].shift(-cfg.horizon) / d[cfg.close_col] - 1.0
    )
    return d


def daily_cs_ic(daily_df: pd.DataFrame, factor_col: str, ret_col: str, min_cs_size: int) -> float:
    x = daily_df[[factor_col, ret_col]].dropna()
    if len(x) < min_cs_size:
        return np.nan
    if x[factor_col].nunique() < 2 or x[ret_col].nunique() < 2:
        return np.nan
    return x[factor_col].corr(x[ret_col], method="spearman")


def _timeseries_ic(df: pd.DataFrame, factor_col: str, ret_col: str, min_ts_size: int) -> pd.Series:
    rows = []
    for sym, g in df.groupby("symbol"):
        pair = g[[factor_col, ret_col]].dropna()
        if len(pair) < min_ts_size:
            continue
        if pair[factor_col].nunique() < 2 or pair[ret_col].nunique() < 2:
            continue
        ic = pair[factor_col].corr(pair[ret_col], method="spearman")
        if pd.notna(ic):
            rows.append((sym, float(ic)))
    out = pd.Series({k: v for k, v in rows}, dtype=float)
    out.name = "ic"
    return out


def calc_ic_series(df: pd.DataFrame, factor_col: str, cfg: EvalConfig, ret_col: str) -> pd.Series:
    if factor_col not in df.columns:
        return pd.Series(dtype=float, name="ic")

    mode = getattr(cfg, "ic_mode", "auto")

    cs_series = (
        df.groupby(cfg.date_col)
        .apply(lambda g: daily_cs_ic(g, factor_col, ret_col, cfg.min_cs_size))
        .dropna()
    )
    cs_series.name = "ic"

    if mode == "cs":
        return cs_series

    ts_series = _timeseries_ic(df, factor_col, ret_col, getattr(cfg, "min_ts_size", 60))
    if mode == "ts":
        return ts_series

    # auto: prefer CS when enough coverage, otherwise fallback to TS
    if len(cs_series) >= 20:
        return cs_series
    return ts_series if len(ts_series) > 0 else cs_series


def summarize_ic(ic_series: pd.Series) -> dict:
    n = len(ic_series)
    if n == 0:
        return dict(ic_mean=np.nan, ic_std=np.nan, ic_ir=np.nan, ic_t=np.nan, ic_p=np.nan, ic_n=0)

    mu = ic_series.mean()
    sd = ic_series.std(ddof=1)
    ic_ir = mu / sd if sd and sd > 0 else np.nan
    ic_t = mu / (sd / np.sqrt(n)) if sd and sd > 0 else np.nan
    ic_p = stats.ttest_1samp(ic_series.values, 0.0).pvalue if n > 1 else np.nan
    return dict(ic_mean=mu, ic_std=sd, ic_ir=ic_ir, ic_t=ic_t, ic_p=ic_p, ic_n=n)


def assign_quantile_labels(s: pd.Series, q: int) -> pd.Series:
    valid = s.dropna()
    if valid.nunique() < min(3, q):
        return pd.Series(index=s.index, dtype=float)
    try:
        labels = pd.qcut(valid.rank(method="first"), q=q, labels=False) + 1
    except Exception:
        return pd.Series(index=s.index, dtype=float)
    out = pd.Series(index=s.index, dtype=float)
    out.loc[valid.index] = labels.astype(float)
    return out


def calc_quantile_returns(df: pd.DataFrame, factor_col: str, cfg: EvalConfig, ret_col: str) -> pd.DataFrame:
    if factor_col not in df.columns:
        return pd.DataFrame()

    d = df[[cfg.date_col, cfg.symbol_col, factor_col, ret_col]].copy()
    d["q"] = d.groupby(cfg.date_col)[factor_col].transform(lambda s: assign_quantile_labels(s, cfg.n_quantiles))
    d = d.dropna(subset=["q", ret_col]).copy()
    if d.empty:
        return pd.DataFrame()
    d["q"] = d["q"].astype(int)

    return d.groupby([cfg.date_col, "q"])[ret_col].mean().unstack("q").sort_index(axis=1)


def monotonicity_score(qret_mean: pd.Series) -> float:
    qret_mean = qret_mean.dropna()
    if len(qret_mean) < 3:
        return np.nan
    x = np.arange(1, len(qret_mean) + 1)
    corr, _ = stats.spearmanr(x, qret_mean.values)
    return corr


def summarize_layer(qret: pd.DataFrame) -> dict:
    if qret.empty:
        return {
            "qret_mean_dict": {},
            "ls_mean": np.nan,
            "ls_t": np.nan,
            "ls_p": np.nan,
            "mono_score": np.nan,
            "ls_n": 0,
        }

    qret_mean = qret.mean(axis=0)
    q_cols = sorted(qret.columns.tolist())
    q_low, q_high = q_cols[0], q_cols[-1]
    ls = (qret[q_high] - qret[q_low]).dropna()

    if len(ls) > 1 and ls.std(ddof=1) > 0:
        ls_t = ls.mean() / (ls.std(ddof=1) / np.sqrt(len(ls)))
        ls_p = stats.ttest_1samp(ls.values, 0.0).pvalue
    else:
        ls_t, ls_p = np.nan, np.nan

    return {
        "qret_mean_dict": {f"Q{int(k)}": v for k, v in qret_mean.items()},
        "ls_mean": ls.mean() if len(ls) > 0 else np.nan,
        "ls_t": ls_t,
        "ls_p": ls_p,
        "mono_score": monotonicity_score(qret_mean),
        "ls_n": int(len(ls)),
    }


def recommend_factor(ic_mean, ic_t, ls_t, mono, cfg: EvalConfig, ic_n: int = 0) -> str:
    if ic_n <= 0 or pd.isna(ic_mean):
        return "INSUFFICIENT_DATA"

    keep_cond = (
        (pd.notna(ic_mean) and abs(ic_mean) >= cfg.ic_mean_keep)
        and (pd.notna(ic_t) and abs(ic_t) >= cfg.ic_t_keep)
        and (pd.notna(ls_t) and abs(ls_t) >= cfg.ls_t_keep)
        and (pd.notna(mono) and abs(mono) >= cfg.mono_keep)
    )
    if keep_cond:
        return "KEEP"

    watch_cond = (
        (pd.notna(ic_mean) and abs(ic_mean) >= cfg.ic_mean_watch)
        and (pd.notna(ic_t) and abs(ic_t) >= cfg.ic_t_watch)
        and (pd.notna(ls_t) and abs(ls_t) >= cfg.ls_t_watch)
        and (pd.notna(mono) and abs(mono) >= cfg.mono_watch)
    )
    if watch_cond:
        return "WATCH"
    return "DROP"


def _empty_summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "factor",
            "ic_mean",
            "ic_std",
            "ic_ir",
            "ic_t",
            "ic_p",
            "ic_n",
            "ls_mean_qhi_qlo",
            "ls_t",
            "ls_p",
            "ls_n",
            "mono_score",
            "recommendation",
            "ic_mode_used",
        ]
    )


def evaluate_factors(df: pd.DataFrame, factor_cols: list[str], cfg: EvalConfig = EvalConfig()):
    d = add_forward_return(df, cfg)
    ret_col = f"fwd_ret_{cfg.horizon}"

    rows: list[dict] = []
    qret_dict: dict[str, pd.DataFrame] = {}
    ic_dict: dict[str, pd.Series] = {}

    for f in factor_cols:
        if f not in d.columns:
            continue

        ic_series = calc_ic_series(d, f, cfg, ret_col)
        ic_stat = summarize_ic(ic_series)

        qret = calc_quantile_returns(d, f, cfg, ret_col)
        layer_stat = summarize_layer(qret)

        mode = getattr(cfg, "ic_mode", "auto")
        rec = recommend_factor(
            ic_mean=ic_stat["ic_mean"],
            ic_t=ic_stat["ic_t"],
            ls_t=layer_stat["ls_t"],
            mono=layer_stat["mono_score"],
            cfg=cfg,
            ic_n=ic_stat["ic_n"],
        )

        row = {
            "factor": f,
            "ic_mean": ic_stat["ic_mean"],
            "ic_std": ic_stat["ic_std"],
            "ic_ir": ic_stat["ic_ir"],
            "ic_t": ic_stat["ic_t"],
            "ic_p": ic_stat["ic_p"],
            "ic_n": ic_stat["ic_n"],
            "ls_mean_qhi_qlo": layer_stat["ls_mean"],
            "ls_t": layer_stat["ls_t"],
            "ls_p": layer_stat["ls_p"],
            "ls_n": layer_stat["ls_n"],
            "mono_score": layer_stat["mono_score"],
            "recommendation": rec,
            "ic_mode_used": mode,
        }

        for k, v in layer_stat["qret_mean_dict"].items():
            row[f"{k}_mean_ret"] = v

        rows.append(row)
        qret_dict[f] = qret
        ic_dict[f] = ic_series

    if not rows:
        return _empty_summary_df(), qret_dict, ic_dict

    summary_df = pd.DataFrame(rows)

    rank_map = {"KEEP": 0, "WATCH": 1, "DROP": 2, "INSUFFICIENT_DATA": 3}
    summary_df["_rk"] = summary_df["recommendation"].map(rank_map).fillna(9)
    summary_df["_abs_ic_t"] = summary_df["ic_t"].abs()
    summary_df = summary_df.sort_values(["_rk", "_abs_ic_t"], ascending=[True, False]).drop(
        columns=["_rk", "_abs_ic_t"]
    )
    summary_df = summary_df.reset_index(drop=True)

    return summary_df, qret_dict, ic_dict


def print_factor_report(summary_df: pd.DataFrame, topn: int = 30):
    show_cols = [
        "factor",
        "recommendation",
        "ic_mean",
        "ic_t",
        "ic_ir",
        "ic_n",
        "ls_mean_qhi_qlo",
        "ls_t",
        "ls_n",
        "mono_score",
    ]
    cols = [c for c in show_cols if c in summary_df.columns]
    print("=== 因子评估汇总（Top）===")
    if summary_df.empty:
        print("<empty>")
        return
    print(summary_df[cols].head(topn).to_string(index=False))
