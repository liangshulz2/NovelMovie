"""Portfolio sizing engine."""

import numpy as np


def volatility_target(vol: float, target: float) -> float:
    """Convert volatility estimate to a base position under target-vol scaling."""
    if vol <= 0:
        return 0.0
    return target / vol


def kelly_scale(alpha: float, vol: float) -> float:
    if vol <= 0:
        return 0.0
    k = alpha / (vol**2)
    return float(np.clip(k, -1, 1))


def final_position(signal: int, base: float, kelly: float, max_position: float) -> float:
    pos = signal * base * kelly
    return float(np.clip(pos, -max_position, max_position))
