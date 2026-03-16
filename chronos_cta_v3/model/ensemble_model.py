"""Model ensembling for alpha fusion."""

import numpy as np


def ensemble_alpha(chronos_pred: np.ndarray, ml_pred: float, w_chronos: float = 0.6) -> float:
    chronos_ret = (chronos_pred.mean() - chronos_pred[0]) / chronos_pred[0]
    return float(w_chronos * chronos_ret + (1 - w_chronos) * ml_pred)
