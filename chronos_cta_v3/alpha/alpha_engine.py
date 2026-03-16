"""Alpha to signal translation."""


def generate_signal(alpha: float, threshold: float = 0.01) -> int:
    if alpha > threshold:
        return 1
    if alpha < -threshold:
        return -1
    return 0
