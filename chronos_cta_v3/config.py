"""Global configuration for Chronos CTA V3."""

SYMBOLS = [
    "RB0",
    "SA0",
    "FG0",
    "CU0",
    "AL0",
    "M0",
    "Y0",
]

MODEL_PATH = "E:/ai/chronos-2"
DEVICE = "cpu"
PRED_LEN = 10

TARGET_VOL = 0.15
MAX_POSITION = 0.2
MAX_PORTFOLIO_RISK = 0.35

CAPITAL = 100000

# Data / model defaults
LOOKBACK = 260
TRAIN_WINDOW = 360
MIN_HISTORY = 120
SEED = 42
