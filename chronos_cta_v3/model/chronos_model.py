"""Chronos model wrapper."""

import numpy as np
import torch
from chronos import Chronos2Pipeline


class ChronosModel:
    def __init__(self, path: str, device: str = "cpu"):
        self.model = Chronos2Pipeline.from_pretrained(path, device_map=device)

    def predict(self, close, factors, pred_len: int = 10) -> np.ndarray:
        target = torch.tensor(close.values.astype(np.float32))
        cov = torch.tensor(factors.values.astype(np.float32))

        forecast = self.model.predict(
            inputs=[{"target": target, "past_covariates": cov}],
            prediction_length=pred_len,
            context_length=len(close),
        )
        pred = forecast[0].cpu().numpy()
        return np.squeeze(pred, axis=0)
