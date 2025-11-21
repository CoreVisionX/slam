from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from hydra.utils import instantiate
from registration.registration import RectifiedStereoFrame, StereoDepthFrame
import tests.test_utils as test_utils
from slam.hydra_utils import compose_config, extract_target_config


@dataclass
class DepthVariant:
    label: str
    frames: list[StereoDepthFrame]


@dataclass
class PreprocessedSequence:
    sequence: test_utils.FrameSequenceWithGroundTruth
    rectified_frames: list[RectifiedStereoFrame]
    depth_variants: list[DepthVariant]
    label: str
    extras: dict[str, Any]


@dataclass
class PreprocessingState:
    sequence: test_utils.FrameSequenceWithGroundTruth | None = None
    rectified_frames: list[RectifiedStereoFrame] | None = None
    depth_variants: list[DepthVariant] = field(default_factory=list)
    label: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def require_sequence(self) -> test_utils.FrameSequenceWithGroundTruth:
        if self.sequence is None:
            raise RuntimeError("Sequence has not been loaded yet. Ensure a loader stage runs first.")
        return self.sequence

    def require_rectified_frames(self) -> list[RectifiedStereoFrame]:
        if self.rectified_frames is None:
            raise RuntimeError("Rectified frames are unavailable. Add a rectification stage to the pipeline.")
        return self.rectified_frames

    def add_depth_variant(self, variant: DepthVariant) -> None:
        self.depth_variants.append(variant)


class SequencePreprocessor:
    """Executes a configurable sequence of preprocessing stages."""

    def __init__(
        self,
        stages: Sequence[Any],
        default_max_depth: float | None = None,
    ) -> None:
        if not stages:
            raise ValueError("At least one pipeline stage must be provided.")
        self.stages = list(stages)
        self._default_max_depth = default_max_depth

    def prepare(
        self,
        seed: int | None = None,
        max_depth: float | None = None,
    ) -> PreprocessedSequence:
        state = PreprocessingState()
        depth_limit = max_depth if max_depth is not None else self._default_max_depth

        for stage in self.stages:
            stage(
                state=state,
                seed=seed,
                max_depth=depth_limit,
            )

        sequence = state.require_sequence()
        rectified_frames = state.require_rectified_frames()
        label = state.label or "sequence"
        return PreprocessedSequence(
            sequence=sequence,
            rectified_frames=rectified_frames,
            depth_variants=list(state.depth_variants),
            label=label,
            extras=dict(state.extras),
        )


@dataclass
class LoadSequenceStage:
    loader: Any

    def __call__(
        self,
        *,
        state: PreprocessingState,
        seed: int | None,
        max_depth: float | None,
    ) -> None:
        del max_depth  # unused in this stage
        sequence = self.loader.load_sequence(seed=seed)
        state.sequence = sequence
        state.label = self.loader.dataset_label()


@dataclass
class RectifyFramesStage:
    rectifier: Any

    def __call__(
        self,
        *,
        state: PreprocessingState,
        seed: int | None,
        max_depth: float | None,
    ) -> None:
        del seed, max_depth
        sequence = state.require_sequence()
        state.rectified_frames = self.rectifier.rectify(sequence)


@dataclass
class DepthEstimationStage:
    label: str
    estimator: Any

    def __call__(
        self,
        *,
        state: PreprocessingState,
        seed: int | None,
        max_depth: float | None,
    ) -> None:
        del seed
        sequence = state.require_sequence()
        rectified = state.require_rectified_frames()
        depth_frames = self.estimator.build(sequence, rectified, max_depth=max_depth)
        state.add_depth_variant(DepthVariant(label=self.label, frames=depth_frames))


def load_euroc_pipeline(
    config_path: str | Path,
    overrides: Sequence[str] | None = None,
    **override_kwargs: object,
) -> SequencePreprocessor:
    cfg = compose_config(config_path, overrides=overrides, **override_kwargs)
    pipeline_cfg = extract_target_config(cfg, context=str(config_path))
    return instantiate(pipeline_cfg, _convert_="partial")
