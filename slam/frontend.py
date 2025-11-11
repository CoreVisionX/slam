"""Front-end processing for stereo SLAM."""

from __future__ import annotations

from dataclasses import dataclass
import time

from depth.sgbm import SGBM
from registration.registration import (
    FeatureFrame,
    RectifiedStereoFrame,
    StereoFrame,
    StereoDepthFrame,
)
from slam.matcher_factory import FeatureMatcher
from cvx_utils import share_feature_frame


@dataclass(slots=True)
class FrontendTimings:
    rectify: float
    depth: float
    feature_detection: float

    @property
    def total(self) -> float:
        return self.rectify + self.depth + self.feature_detection


@dataclass(slots=True)
class FrontendOutput:
    feature_frame: FeatureFrame
    depth_frame: StereoDepthFrame
    timings: FrontendTimings


class StereoFrontend:
    """Compute depth and features for incoming stereo frames."""

    def __init__(
        self,
        depth_estimator: SGBM,
        feature_detector: FeatureMatcher,
        *,
        rectify_inputs: bool = True,
    ) -> None:
        self._depth_estimator = depth_estimator
        self._feature_detector = feature_detector
        self._rectify_inputs = rectify_inputs

    def process(self, frame: StereoFrame) -> FrontendOutput:
        """Rectify the frame, compute depth, and detect features."""

        rectify_start = time.perf_counter()
        if self._rectify_inputs:
            rectified_frame = frame.rectify()
            rectify_end = time.perf_counter()
            rectify_duration = rectify_end - rectify_start
        else:
            if isinstance(frame, RectifiedStereoFrame):
                rectified_frame = frame
            else:
                rectified_frame = RectifiedStereoFrame(
                    left=frame.left,
                    right=frame.right,
                    left_rect=frame.left,
                    right_rect=frame.right,
                    calibration=frame.calibration,
                )
            rectify_end = time.perf_counter()
            rectify_duration = 0.0

        depth_start = rectify_end
        depth_frame = self._depth_estimator.compute_depth(rectified_frame)
        depth_end = time.perf_counter()

        detect_start = depth_end
        feature_frame = self._feature_detector.detect_features([depth_frame])[0]
        feature_frame = share_feature_frame(feature_frame)
        detect_end = time.perf_counter()

        timings = FrontendTimings(
            rectify=rectify_duration,
            depth=depth_end - depth_start,
            feature_detection=detect_end - detect_start,
        )

        return FrontendOutput(
            feature_frame=feature_frame,
            depth_frame=depth_frame,
            timings=timings,
        )
