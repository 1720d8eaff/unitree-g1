"""Explicit action contract; a GR00T integration must supply its own mapping.

REAL_G1 also contains joint modalities. A 14-DoF fixed-base palm simulator is
not automatically equivalent to that pretrained embodiment. This contract
therefore uses a custom, explicit body-frame EEF delta convention.
"""
from dataclasses import dataclass
import numpy as np
from .geometry import decode_relative


@dataclass
class ActionChunk:
    observation_timestamp: float
    generated_timestamp: float
    sample_dt: float
    relative_eef: np.ndarray  # (T,2,6), relative to ONE observation pose
    frame: str = "eef_body"

    def validate(self, now, max_age=.30):
        a = np.asarray(self.relative_eef)
        if self.frame != "eef_body" or a.ndim != 3 or a.shape[1:] != (2, 6):
            raise ValueError("Required shape (T,2,6) in eef_body frame")
        if not np.isfinite(a).all() or not np.isfinite(self.sample_dt) or self.sample_dt <= 0:
            raise ValueError("Invalid action values or sample period")
        if not np.isfinite([now, self.observation_timestamp, self.generated_timestamp]).all():
            raise ValueError("Nonfinite timestamp")
        if self.generated_timestamp > now + 1e-6 or self.generated_timestamp < self.observation_timestamp:
            raise ValueError("Unordered action timestamps")
        return 0 <= now-self.observation_timestamp <= max_age

    def decode(self, observation_hands):
        if np.asarray(observation_hands).shape != (2, 4, 4):
            raise ValueError("Two observation poses are required")
        return np.array([[decode_relative(h, delta) for h, delta in zip(observation_hands, row)]
                         for row in self.relative_eef])
