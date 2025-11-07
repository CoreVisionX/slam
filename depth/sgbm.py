from __future__ import annotations

from dataclasses import asdict
from typing import Literal, Optional

import cv2
import numpy as np

from registration.registration import FramePair, RectifiedStereoFrame, StereoDepthFrame


class SGBM:
    """OpenCV Semi-Global Block Matching disparity estimator."""

    def __init__(
        self,
        min_disparity: int = 0,
        num_disparities: int = 16 * 6,
        block_size: int = 5,
        pre_filter_cap: int = 63,
        uniqueness_ratio: int = 10,
        speckle_window_size: int = 100,
        speckle_range: int = 32,
        disp12_max_diff: int = 1,
        mode: Optional[int] = None,
        p1: Optional[int] = None,
        p2: Optional[int] = None,
        image_color: Literal["BGR", "RGB"] = "RGB",
    ) -> None:
        if num_disparities % 16 != 0:
            raise ValueError("num_disparities must be divisible by 16.")
        if block_size % 2 == 0 or block_size < 1:
            raise ValueError("block_size must be a positive odd integer.")

        if mode is None:
            mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

        # Default penalties follow the OpenCV docs and depend on the number of channels.
        channels = 1  # images are converted to grayscale before matching
        if p1 is None:
            p1 = 8 * channels * block_size * block_size
        if p2 is None:
            p2 = 32 * channels * block_size * block_size

        self._matcher = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=p1,
            P2=p2,
            preFilterCap=pre_filter_cap,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
            disp12MaxDiff=disp12_max_diff,
            mode=mode,
        )
        self._color_conversion = cv2.COLOR_BGR2GRAY if image_color == "BGR" else cv2.COLOR_RGB2GRAY
        self._min_disparity = min_disparity
        self._scale = 16.0
        self._invalid_raw_disparity = int((min_disparity - 1) * self._scale)

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Compute the disparity map for a pair of rectified stereo images."""
        left_image = self._prepare_image(left)
        right_image = self._prepare_image(right)

        raw_disparity = self._matcher.compute(left_image, right_image)
        disparity = raw_disparity.astype(np.float32) / self._scale
        disparity[raw_disparity <= self._invalid_raw_disparity] = np.nan
        return -disparity

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] == 3:
            array = cv2.cvtColor(array, self._color_conversion)
        elif array.ndim != 2:
            raise ValueError(f"Expected image with 1 or 3 channels, got shape {array.shape}")

        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            max_value = np.max(array) if array.size else 0.0
            if max_value <= 1.0:
                array *= 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(array)

    def compute_depth(self, frame: RectifiedStereoFrame, max_depth: float = 60.0) -> StereoDepthFrame:
        disparity = self(frame.left_rect, frame.right_rect)

        depth_xyz = cv2.reprojectImageTo3D(disparity, frame.calibration.Q)

        depth = depth_xyz[:, :, 2]
        depth[depth > max_depth] = np.nan
    
        return StereoDepthFrame(
            **{k: v for k, v in frame.__dict__.items() if not k.startswith('__')},
            left_depth=depth,
            left_depth_xyz=depth_xyz
        )

    def compute_depth_pair(self, pair: FramePair[RectifiedStereoFrame], max_depth: float = 60.0) -> FramePair[StereoDepthFrame]:
        first_depth = self.compute_depth(pair.first, max_depth)
        second_depth = self.compute_depth(pair.second, max_depth)

        return pair.__class__[StereoDepthFrame](
            **{k: v for k, v in pair.__dict__.items() if not k.startswith('__') and k not in ['first', 'second']},
            first=first_depth,
            second=second_depth
        )
