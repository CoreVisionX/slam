from __future__ import annotations

from typing import Literal, Protocol

from registration.lighterglue import LighterglueMatcher
from registration.orb import OrbMatcher
from registration.registration import FeatureFrame, FramePair, MatchedFramePair, StereoDepthFrame


MatcherType = Literal["lighterglue", "orb"]


class FeatureMatcher(Protocol):
    def detect_features(self, frames: list[StereoDepthFrame]) -> list[FeatureFrame]:
        ...

    def match(self, pairs: list[FramePair[FeatureFrame]]) -> list[MatchedFramePair[FeatureFrame]]:
        ...


def create_matcher(matcher_type: MatcherType, *, num_features: int = 4096) -> FeatureMatcher:
    if matcher_type == "lighterglue":
        return LighterglueMatcher(
            num_features=num_features,
            compile=False,
            device="cuda",
            use_lighterglue_matching=True,
        )
    if matcher_type == "orb":
        return OrbMatcher(num_features=num_features)
    raise ValueError(f"Unsupported matcher type: {matcher_type}")
