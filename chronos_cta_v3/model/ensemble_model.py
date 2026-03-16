"""Model ensembling for alpha fusion."""

from __future__ import annotations

import numpy as np


def ensemble_alpha(chronos_pred: np.ndarray, ml_pred: float, w_chronos: float = 0.6) -> float:
    chronos_ret = (chronos_pred.mean() - chronos_pred[0]) / chronos_pred[0]
    return float(w_chronos * chronos_ret + (1 - w_chronos) * ml_pred)


def bayesian_model_averaging(predictions: dict[str, float], errors: dict[str, float], temp: float = 1.0) -> tuple[float, dict[str, float]]:
    """Bayesian model averaging using inverse error as model likelihood proxy."""
    keys = [k for k in predictions.keys() if k in errors and errors[k] > 0]
    if not keys:
        return 0.0, {}

    log_like = np.array([-errors[k] / max(temp, 1e-8) for k in keys], dtype=float)
    log_like = log_like - log_like.max()
    probs = np.exp(log_like)
    probs = probs / probs.sum()

    alpha = float(np.sum([predictions[k] * probs[i] for i, k in enumerate(keys)]))
    weights = {k: float(probs[i]) for i, k in enumerate(keys)}
    return alpha, weights
