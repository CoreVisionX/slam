from __future__ import annotations

from typing import Any

import numpy as np

from slam.registration.registration import FeatureFrame, RectifiedStereoFrame, StereoDepthFrame


def extract_keypoint_attributes(
    depth_frame: StereoDepthFrame,
    keypoints: np.ndarray,
    max_depth: float,
) -> dict[str, Any]:
    """Sample depth-derived attributes for a set of keypoints."""
    h, w = depth_frame.left_depth.shape
    clipped = np.clip(np.round(keypoints), [0, 0], [w - 1, h - 1]).astype(int)
    rows = clipped[:, 1]
    cols = clipped[:, 0]

    depths = depth_frame.left_depth[rows, cols]
    xyz = depth_frame.left_depth_xyz[rows, cols]
    colors = depth_frame.left_rect[rows, cols]

    valid = np.isfinite(depths) & (depths > 0.0) & (depths <= max_depth)
    filtered = {
        "keypoints": keypoints[valid],
        "keypoints_depth": depths[valid],
        "keypoints_3d": xyz[valid],
        "keypoints_color": colors[valid],
        "valid_mask": valid,
    }
    return filtered


def build_feature_frame(
    depth_frame: StereoDepthFrame,
    attributes: dict[str, Any],
) -> FeatureFrame:
    """Create a FeatureFrame that bundles keypoint data and calibration."""
    features: dict[str, Any] = dict(attributes)
    features["keypoints"] = np.asarray(attributes["keypoints"], dtype=np.float32)
    features["image_size"] = depth_frame.left_rect.shape[:2]
    return FeatureFrame(
        left=None,
        right=None,
        left_rect=None,
        right_rect=None,
        left_depth=None,
        left_depth_xyz=None,
        calibration=depth_frame.calibration,
        features=features,
    )


def make_feature_frame_for_view(
    rectified_frame: RectifiedStereoFrame,
    keypoints: np.ndarray,
) -> FeatureFrame:
    """Create a FeatureFrame-like wrapper for a rectified stereo view."""
    features: dict[str, Any] = {
        "keypoints": np.asarray(keypoints, dtype=np.float32),
        "image_size": rectified_frame.left_rect.shape[:2],
    }
    return FeatureFrame(
        left=None,
        right=None,
        left_rect=None,
        right_rect=None,
        left_depth=None,
        left_depth_xyz=None,
        calibration=rectified_frame.calibration,
        features=features,
    )


__all__ = [
    "build_feature_frame",
    "extract_keypoint_attributes",
    "make_feature_frame_for_view",
]
