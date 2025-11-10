"""Utilities for working with physical stereo rigs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from registration.registration import StereoCalibration

_CALIBRATION_REQUIRED_KEYS: tuple[str, ...] = (
    "K1",
    "D1",
    "K2",
    "D2",
    "R",
    "T",
    "R1",
    "R2",
    "P1",
    "P2",
    "Q",
    "image_size",
)


def load_stereo_calibration_npz(calib_path: str | Path) -> StereoCalibration:
    """Load a StereoCalibration object from a calibration npz produced by stereo_calibrate.py."""

    path = Path(calib_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file '{path}' does not exist.")

    with np.load(path, allow_pickle=True) as data:
        missing = _missing_keys(data.files, _CALIBRATION_REQUIRED_KEYS)
        if missing:
            raise KeyError(
                f"Calibration file '{path}' is missing required keys: {', '.join(sorted(missing))}"
            )

        K_left = np.asarray(data["K1"], dtype=np.float64)
        D_left = np.asarray(data["D1"], dtype=np.float64)
        K_right = np.asarray(data["K2"], dtype=np.float64)
        D_right = np.asarray(data["D2"], dtype=np.float64)
        R = np.asarray(data["R"], dtype=np.float64)
        T = np.asarray(data["T"], dtype=np.float64).reshape(3, 1)
        R_left = np.asarray(data["R1"], dtype=np.float64)
        R_right = np.asarray(data["R2"], dtype=np.float64)
        P_left = np.asarray(data["P1"], dtype=np.float64)
        P_right = np.asarray(data["P2"], dtype=np.float64)
        Q = np.asarray(data["Q"], dtype=np.float64)
        size = tuple(int(v) for v in np.asarray(data["image_size"]).ravel())

    if len(size) != 2:
        raise ValueError(f"Expected image_size to have two entries, got {size}.")

    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image_size {size}; width and height must be positive.")

    image_size = (width, height)
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left, D_left, R_left, P_left, image_size, cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, D_right, R_right, P_right, image_size, cv2.CV_32FC1
    )

    return StereoCalibration(
        K_left=K_left,
        K_right=K_right,
        K_left_rect=P_left[:3, :3],
        K_right_rect=P_right[:3, :3],
        D_left=D_left,
        D_right=D_right,
        R=R,
        T=T,
        R_left=R_left,
        R_right=R_right,
        P_left=P_left,
        P_right=P_right,
        Q=Q,
        map_left_x=map_left_x,
        map_left_y=map_left_y,
        map_right_x=map_right_x,
        map_right_y=map_right_y,
        width=width,
        height=height,
    )


def split_wide_frame(
    frame: np.ndarray,
    *,
    mode: str = "half",
    split_px: int | None = None,
    ratio: float = 0.5,
    swap_halves: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a single wide stereo frame into left and right views."""

    if frame.ndim < 2:
        raise ValueError("Expected frame with height and width dimensions.")

    height, width = frame.shape[:2]
    mode_normalized = mode.lower()
    if mode_normalized == "half":
        column = width // 2
    elif mode_normalized == "px":
        if split_px is None:
            raise ValueError("split_px must be provided when mode='px'.")
        column = int(split_px)
    elif mode_normalized == "ratio":
        column = int(round(width * float(ratio)))
    else:
        raise ValueError(f"Unknown split mode '{mode}'. Expected 'half', 'px', or 'ratio'.")

    column = max(1, min(width - 1, column))
    left = frame[:, :column]
    right = frame[:, column:]

    if swap_halves:
        left, right = right, left

    if left.size == 0 or right.size == 0:
        raise ValueError(
            f"Split produced an empty image (mode={mode}, column={column}); "
            "check camera resolution and split arguments."
        )

    return left, right


def _missing_keys(actual: Iterable[str], expected: Iterable[str]) -> set[str]:
    actual_set = {key for key in actual}
    return {key for key in expected if key not in actual_set}


__all__ = ["load_stereo_calibration_npz", "split_wide_frame"]
