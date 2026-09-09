import numpy as np


class EMA:
    """Exponential Moving Average — same pattern as Gesture-Media-control baseline (alpha=0.25)."""

    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = np.array(new_value, dtype=float) if hasattr(new_value, '__len__') else float(new_value)
        else:
            self.value = self.alpha * np.array(new_value, dtype=float) + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class QuatEMA:
    """EMA for quaternions — lerps component-wise then renormalises."""

    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self.value = np.array([0.0, 0.0, 0.0, 1.0])  # identity [x,y,z,w]

    def update(self, q: np.ndarray) -> np.ndarray:
        # Ensure same hemisphere (avoid double-cover flip)
        if np.dot(self.value, q) < 0:
            q = -q
        self.value = self.alpha * q + (1 - self.alpha) * self.value
        norm = np.linalg.norm(self.value)
        if norm > 1e-6:
            self.value /= norm
        return self.value.copy()

    def reset(self):
        self.value = np.array([0.0, 0.0, 0.0, 1.0])
