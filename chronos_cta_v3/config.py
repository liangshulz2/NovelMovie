"""Global configuration for Chronos CTA V3."""

from __future__ import annotations

import os

SYMBOLS = [
    "RB0",
    "SA0",
    "FG0",
    "CU0",
    "AL0",
    "M0",
    "Y0",
]

MODEL_PATH = os.getenv("CHRONOS_MODEL_PATH", "E:/ai/chronos-2")
DEVICE = os.getenv("CHRONOS_DEVICE", "cpu")
PRED_LEN = int(os.getenv("CHRONOS_PRED_LEN", "10"))

TARGET_VOL = float(os.getenv("CHRONOS_TARGET_VOL", "0.15"))
MAX_POSITION = float(os.getenv("CHRONOS_MAX_POSITION", "0.2"))
MAX_PORTFOLIO_RISK = float(os.getenv("CHRONOS_MAX_PORTFOLIO_RISK", "0.35"))

CAPITAL = float(os.getenv("CHRONOS_CAPITAL", "100000"))

LOOKBACK = int(os.getenv("CHRONOS_LOOKBACK", "260"))
TRAIN_WINDOW = int(os.getenv("CHRONOS_TRAIN_WINDOW", "360"))
MIN_HISTORY = int(os.getenv("CHRONOS_MIN_HISTORY", "120"))
SEED = int(os.getenv("CHRONOS_SEED", "42"))
