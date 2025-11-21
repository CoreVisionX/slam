from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from depth.sgbm import SGBM
from registration.registration import (
    RectifiedStereoFrame,
    StereoCalibration,
    StereoDepthFrame,
    StereoFrame,
)
import tests.test_utils as test_utils


def _remap_ground_truth_depth(
    raw_depth: np.ndarray,
    calibration: StereoCalibration,
) -> np.ndarray:
    depth = np.asarray(raw_depth, dtype=np.float32)
    expected_shape = (calibration.height, calibration.width)
    if depth.shape != expected_shape:
        raise ValueError(
            f"Depth map shape {depth.shape} does not match calibration size {expected_shape}."
        )
    remapped = cv2.remap(
        depth,
        calibration.map_left_x,
        calibration.map_left_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return remapped


def _depth_map_to_xyz(
    depth_map: np.ndarray,
    calibration_matrix: np.ndarray,
) -> np.ndarray:
    h, w = depth_map.shape
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)
    fx = float(calibration_matrix[0, 0])
    fy = float(calibration_matrix[1, 1])
    cx = float(calibration_matrix[0, 2])
    cy = float(calibration_matrix[1, 2])
    if np.isclose(fx, 0.0) or np.isclose(fy, 0.0):
        raise ValueError("Invalid rectified intrinsics; fx/fy must be non-zero.")
    z = depth_map
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    xyz = np.stack((x, y, z), axis=-1).astype(np.float32, copy=False)
    invalid = ~np.isfinite(z)
    xyz[invalid] = np.nan
    return xyz


def _build_ground_truth_depth_frame(
    rectified_frame: RectifiedStereoFrame,
    raw_depth_map: np.ndarray,
    max_depth: float,
) -> StereoDepthFrame:
    depth_rect = _remap_ground_truth_depth(raw_depth_map, rectified_frame.calibration)
    depth_rect = depth_rect.astype(np.float32, copy=False)
    invalid = (~np.isfinite(depth_rect)) | (depth_rect <= 0.0) | (depth_rect > max_depth)
    depth_rect[invalid] = np.nan
    depth_xyz = _depth_map_to_xyz(depth_rect, rectified_frame.calibration.K_left_rect)
    depth_xyz[invalid, :] = np.nan
    return StereoDepthFrame(
        left=rectified_frame.left,
        right=rectified_frame.right,
        calibration=rectified_frame.calibration,
        left_rect=rectified_frame.left_rect,
        right_rect=rectified_frame.right_rect,
        left_depth=depth_rect,
        left_depth_xyz=depth_xyz,
    )


@dataclass
class GroundTruthDepthEstimator:
    """Depth estimator that consumes dataset-provided depth maps."""

    default_max_depth: float = 80.0

    def build(
        self,
        sequence: test_utils.FrameSequenceWithGroundTruth[StereoFrame],
        rectified_frames: list[RectifiedStereoFrame],
        max_depth: float | None = None,
    ) -> list[StereoDepthFrame]:
        depth_maps = sequence.ground_truth_depths
        if depth_maps is None:
            raise RuntimeError("Ground-truth depth requested but not provided by the sequence loader.")
        if len(rectified_frames) != len(depth_maps):
            raise ValueError("Depth map count does not match the number of rectified frames.")
        limit = max_depth if max_depth is not None else self.default_max_depth
        return [
            _build_ground_truth_depth_frame(rectified_frame, raw_depth, limit)
            for rectified_frame, raw_depth in zip(rectified_frames, depth_maps)
        ]


@dataclass
class SGBMDepthEstimator:
    """Estimates depth using OpenCV's SGBM implementation."""

    min_disparity: int = 0
    num_disparities: int = 16 * 4
    block_size: int = 5
    pre_filter_cap: int = 63
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 32
    disp12_max_diff: int = 1
    mode: int | None = None
    p1: int | None = None
    p2: int | None = None
    image_color: str = "RGB"
    default_max_depth: float = 60.0

    def __post_init__(self) -> None:
        self._matcher = SGBM(
            min_disparity=self.min_disparity,
            num_disparities=self.num_disparities,
            block_size=self.block_size,
            pre_filter_cap=self.pre_filter_cap,
            uniqueness_ratio=self.uniqueness_ratio,
            speckle_window_size=self.speckle_window_size,
            speckle_range=self.speckle_range,
            disp12_max_diff=self.disp12_max_diff,
            mode=self.mode,
            p1=self.p1,
            p2=self.p2,
            image_color=self.image_color,  # type: ignore[arg-type]
        )

    def build(
        self,
        sequence: test_utils.FrameSequenceWithGroundTruth[StereoFrame],
        rectified_frames: list[RectifiedStereoFrame],
        max_depth: float | None = None,
    ) -> list[StereoDepthFrame]:
        del sequence  # unused but kept for symmetry with other estimators
        limit = max_depth if max_depth is not None else self.default_max_depth
        return [self._matcher.compute_depth(frame, max_depth=limit) for frame in rectified_frames]
