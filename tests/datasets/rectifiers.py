from __future__ import annotations

from dataclasses import dataclass

from registration.registration import RectifiedStereoFrame, StereoFrame
import tests.test_utils as test_utils


@dataclass
class StereoSequenceRectifier:
    """Rectifies stereo frames when necessary."""

    force_rectification: bool = False

    def rectify(
        self,
        sequence: test_utils.FrameSequenceWithGroundTruth[StereoFrame],
    ) -> list[RectifiedStereoFrame]:
        rectified_frames: list[RectifiedStereoFrame] = []
        for frame in sequence.frames:
            if isinstance(frame, RectifiedStereoFrame) and not self.force_rectification:
                rectified_frames.append(frame)
            else:
                rectified_frames.append(frame.rectify())
        return rectified_frames
