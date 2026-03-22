import numpy as np
import pandas as pd

from chronos_cta_v3.factors.factor_selector import secondary_feature_filter


def _build_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    f1 = rng.normal(size=n)
    f2 = 0.995 * f1 + rng.normal(scale=0.02, size=n)  # highly collinear with f1
    f3 = rng.normal(size=n)
    noise = rng.normal(scale=0.3, size=n)
    target = 0.7 * f1 - 0.5 * f3 + noise
    return pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "target": target})


def test_secondary_filter_removes_highly_correlated_duplicate():
    df = _build_df()
    selected = secondary_feature_filter(
        df=df,
        target_col="target",
        ranked_features=["f1", "f2", "f3"],
        corr_threshold=0.9,
        vif_threshold=20.0,
        max_features=3,
    )

    assert "f1" in selected
    assert "f3" in selected
    # f2 should be dropped due to very high pairwise correlation with f1.
    assert "f2" not in selected


def test_secondary_filter_respects_max_features_and_order_priority():
    df = _build_df()
    selected = secondary_feature_filter(
        df=df,
        target_col="target",
        ranked_features=["f1", "f3", "f2"],
        corr_threshold=0.99,
        vif_threshold=100.0,
        max_features=2,
    )

    assert selected == ["f1", "f3"]
