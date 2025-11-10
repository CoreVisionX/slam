"""Lightweight stereo vision odometry helper for SLAM runners."""

from __future__ import annotations

from dataclasses import dataclass

import gtsam

from depth.sgbm import SGBM
from registration.registration import (
    FeatureFrame,
    FramePair,
    RectifiedStereoFrame,
    StereoDepthFrame,
    StereoFrame,
)
from registration.utils import solve_pnp
from slam.matcher_factory import MatcherType, create_matcher


@dataclass(slots=True)
class OdometryEstimate:
    pose: gtsam.Pose3 | None
    match_count: int
    inlier_count: int
    failure_reason: str | None = None


class VisionOdometryEstimator:
    """Estimate frame-to-frame odometry via feature matching + PnP."""

    def __init__(self, matcher_type: MatcherType, *, rectify_inputs: bool, max_depth_meters: float = 60.0) -> None:
        self._matcher = create_matcher(matcher_type)
        self._depth_estimator = SGBM(num_disparities=16 * 4, block_size=5, image_color="RGB")
        self._rectify_inputs = rectify_inputs
        self._max_depth_meters = max_depth_meters
        self._prev_feature_frame: FeatureFrame | None = None
        self._last_depth_frame: StereoDepthFrame | None = None

    def prime(self, frame: StereoFrame) -> None:
        """Seed the estimator with the very first frame of the sequence."""
        self._prev_feature_frame = self._compute_feature_frame(frame)

    def estimate(self, frame: StereoFrame) -> OdometryEstimate:
        feature_frame = self._compute_feature_frame(frame)
        estimate = OdometryEstimate(pose=None, match_count=0, inlier_count=0)

        if self._prev_feature_frame is not None:
            pair = FramePair(first=self._prev_feature_frame, second=feature_frame)
            matched_pair = self._matcher.match([pair])[0]
            match_count = matched_pair.matches.shape[0]
            pose: gtsam.Pose3 | None = None
            inlier_count = 0
            failure_reason: str | None = None

            if match_count >= 4:
                try:
                    pose, inlier_pair = solve_pnp(matched_pair)
                    inlier_count = inlier_pair.matches.shape[0]
                except Exception as exc:  # noqa: BLE001
                    failure_reason = str(exc)
            else:
                failure_reason = "insufficient_matches"

            estimate = OdometryEstimate(
                pose=pose,
                match_count=match_count,
                inlier_count=inlier_count,
                failure_reason=failure_reason,
            )

        self._restore_feature_storage(feature_frame)
        self._prev_feature_frame = feature_frame
        return estimate

    def _compute_feature_frame(self, frame: StereoFrame) -> FeatureFrame:
        if self._rectify_inputs:
            rectified_frame = frame.rectify()
        elif isinstance(frame, RectifiedStereoFrame):
            rectified_frame = frame
        else:
            rectified_frame = RectifiedStereoFrame(
                left=frame.left,
                right=frame.right,
                left_rect=frame.left,
                right_rect=frame.right,
                calibration=frame.calibration,
            )

        depth_frame = self._depth_estimator.compute_depth(rectified_frame, max_depth=self._max_depth_meters)
        self._last_depth_frame = depth_frame
        feature_frame = self._matcher.detect_features([depth_frame])[0]
        return feature_frame

    @staticmethod
    def _restore_feature_storage(frame: FeatureFrame | None) -> None:
        if frame is None:
            return

        for key in ("keypoints", "descriptors"):
            value = frame.features.get(key)
            if value is None:
                continue
            if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
                frame.features[key] = value.detach().cpu().numpy()

    @property
    def latest_depth_frame(self) -> StereoDepthFrame | None:
        return self._last_depth_frame


__all__ = ["OdometryEstimate", "VisionOdometryEstimator"]
