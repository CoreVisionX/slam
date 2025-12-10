from dataclasses import dataclass
from typing import Sequence

import numpy as np
import gtsam
from pydantic.dataclasses import dataclass as pydantic_dataclass
from slam.registration.registration import StereoCalibration


def _vec_to_list(vec: Sequence[float] | np.ndarray, name: str) -> list[float]:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must be length 3, got shape {arr.shape}")
    return arr.tolist()


def _mat_to_list(mat: Sequence[Sequence[float]] | np.ndarray, name: str) -> list[list[float]]:
    arr = np.asarray(mat, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3, got shape {arr.shape}")
    return arr.tolist()


@pydantic_dataclass
class VIOEstimate:
    timestamp: float
    t: list[float]
    R: list[list[float]]
    v: list[float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _vec_to_list(self.t, "t"))
        object.__setattr__(self, "v", _vec_to_list(self.v, "v"))
        object.__setattr__(self, "R", _mat_to_list(self.R, "R"))

    @classmethod
    def from_numpy(
        cls,
        timestamp: float,
        t: np.ndarray,
        R: np.ndarray,
        v: np.ndarray,
    ) -> "VIOEstimate":
        return cls(timestamp=timestamp, t=t, R=R, v=v)

    def t_np(self) -> np.ndarray:
        return np.asarray(self.t, dtype=float)

    def R_np(self) -> np.ndarray:
        return np.asarray(self.R, dtype=float)

    def v_np(self) -> np.ndarray:
        return np.asarray(self.v, dtype=float)


@dataclass
class VIOCalibration(StereoCalibration):
    imu_from_left: gtsam.Pose3
    imu_from_right: gtsam.Pose3
