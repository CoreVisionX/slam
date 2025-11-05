from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import dataclass
from multiprocessing import shared_memory

import gtsam
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from registration.registration import FeatureFrame


class SharedNDArray:
    """Process-safe view of a NumPy array backed by shared memory."""

    def __init__(
        self,
        array: np.ndarray | None = None,
        *,
        name: str | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: str | np.dtype | None = None,
    ) -> None:
        if array is not None:
            array = np.ascontiguousarray(array)

            if array.nbytes == 0:
                raise ValueError("SharedNDArray requires a non-empty array")

            self.shape = array.shape
            self.dtype = array.dtype
            self._shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
            self.shm_name = self._shm.name
            self._array = np.ndarray(self.shape, dtype=self.dtype, buffer=self._shm.buf)
            self._array[:] = array
        elif name is not None and shape is not None and dtype is not None:
            self.shape = tuple(shape)
            self.dtype = np.dtype(dtype)
            self.shm_name = name
            self._shm = shared_memory.SharedMemory(name=self.shm_name)
            self._array = np.ndarray(self.shape, dtype=self.dtype, buffer=self._shm.buf)
        else:
            raise ValueError("Either array or shared memory metadata must be provided")

    def close(self) -> None:
        if hasattr(self, "_shm") and self._shm is not None:
            self._shm.close()
            self._shm = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except FileNotFoundError:
            pass

    def __array__(self) -> np.ndarray:
        return self._array

    def __len__(self) -> int:
        return len(self._array)

    def __getitem__(self, item):
        return self._array[item]

    def __setitem__(self, key, value) -> None:
        self._array[key] = value

    def __getattr__(self, name):
        return getattr(self._array, name)

    def to_numpy(self) -> np.ndarray:
        return self._array

    def __getstate__(self) -> dict[str, object]:
        return {
            "name": self.shm_name,
            "shape": self.shape,
            "dtype": self.dtype.str,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.shape = tuple(state["shape"])
        self.dtype = np.dtype(state["dtype"])
        self.shm_name = str(state["name"])
        self._shm = shared_memory.SharedMemory(name=self.shm_name)
        self._array = np.ndarray(self.shape, dtype=self.dtype, buffer=self._shm.buf)


@dataclass(slots=True)
class LightweightCalibration:
    K_left: np.ndarray
    K_right: np.ndarray
    K_left_rect: np.ndarray
    K_right_rect: np.ndarray
    D_left: np.ndarray
    D_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    R_left: np.ndarray
    R_right: np.ndarray
    P_left: np.ndarray
    P_right: np.ndarray
    Q: np.ndarray | None
    width: int
    height: int
    map_left_x: None = None
    map_left_y: None = None
    map_right_x: None = None
    map_right_y: None = None

    @classmethod
    def from_calibration(cls, calibration) -> "LightweightCalibration":
        return cls(
            K_left=np.array(calibration.K_left, copy=True),
            K_right=np.array(calibration.K_right, copy=True),
            K_left_rect=np.array(calibration.K_left_rect, copy=True),
            K_right_rect=np.array(calibration.K_right_rect, copy=True),
            D_left=np.array(calibration.D_left, copy=True),
            D_right=np.array(calibration.D_right, copy=True),
            R=np.array(calibration.R, copy=True),
            T=np.array(calibration.T, copy=True),
            R_left=np.array(calibration.R_left, copy=True),
            R_right=np.array(calibration.R_right, copy=True),
            P_left=np.array(calibration.P_left, copy=True),
            P_right=np.array(calibration.P_right, copy=True),
            Q=np.array(calibration.Q, copy=True) if getattr(calibration, "Q", None) is not None else None,
            width=int(calibration.width),
            height=int(calibration.height),
        )


def share_feature_frame(frame: "FeatureFrame") -> "FeatureFrame":
    """Back feature arrays with shared memory for cheaper process transfer."""

    for key, value in list(frame.features.items()):
        if isinstance(value, SharedNDArray):
            continue
        if isinstance(value, np.ndarray) and value.nbytes > 0:
            frame.features[key] = SharedNDArray(value)

    if getattr(frame, "calibration", None) is not None and not isinstance(frame.calibration, LightweightCalibration):
        frame.calibration = LightweightCalibration.from_calibration(frame.calibration)

    return frame


def se3_to_pose3(se3: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(se3[:3, :3]), gtsam.Point3(se3[:3, 3]))

def se3_flattened_to_pose3(se3_flattened: np.ndarray) -> gtsam.Pose3:
    se3 = se3_flattened.reshape(3, 4)

    T = np.eye(4)
    T[:3, :3] = se3[:3, :3]
    T[:3, 3] = se3[:3, 3]

    return se3_to_pose3(T)

def convert_coordinate_frame(pose: gtsam.Pose3, old_to_new_frame: np.ndarray) -> gtsam.Pose3:
    new_R = old_to_new_frame @ pose.rotation().matrix() @ old_to_new_frame.T
    new_t = old_to_new_frame @ pose.translation()

    return gtsam.Pose3(gtsam.Rot3(new_R), gtsam.Point3(new_t))
