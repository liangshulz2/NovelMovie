"""Chronos model wrapper."""

from pathlib import Path

import numpy as np
import torch
from chronos import Chronos2Pipeline


class ChronosModel:
    def __init__(self, path: str, device: str = "cpu"):
        model_path = Path(path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(
                "Chronos model path does not exist: "
                f"{model_path}. Please set CHRONOS_MODEL_PATH to a valid local model directory."
            )

        try:
            self.model = Chronos2Pipeline.from_pretrained(str(model_path), device_map=device)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize Chronos2Pipeline from path "
                f"{model_path} on device '{device}'. Verify model files, runtime device, "
                "and CHRONOS_MODEL_PATH/CHRONOS_DEVICE configuration."
            ) from exc

    def predict(self, close, factors, pred_len: int = 10) -> np.ndarray:
        target = torch.tensor(close.values.astype(np.float32))
        #cov = torch.tensor(factors.values.astype(np.float32))
        cov = {
        col: torch.tensor(factors[col].values.astype(np.float32))
        for col in factors.columns if col not in ["date","close","high","low"]
    }

        forecast = self.model.predict(
            inputs=[{"target": target, "past_covariates": cov}],
            prediction_length=pred_len,
            context_length=len(close),
        )
        pred = forecast[0].cpu().numpy()
        return np.squeeze(pred, axis=0)
