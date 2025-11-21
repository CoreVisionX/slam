from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from slam.registration.registration import RectifiedStereoFrame
import tests.test_utils as test_utils


@dataclass
class EurocSequenceLoader:
    """Loads a EuRoC V1/V2 sequence segment."""

    data_root: str | Path | None = None
    seq_name: str = "MH_01_easy"
    sequence_length: int = 6000
    base_seed: int = 0
    sampling_mode: str = "contiguous"
    min_stride: int = 1
    max_stride: int | None = None
    add_imu_noise: bool = False
    imu_noise_seed: int = 0
    imu_noise_tau_bias: float = 500.0
    alpha: float = 0.0
    resize_to: tuple[int, int] | None = None

    def load_sequence(
        self,
        seed: int | None = None,
    ) -> test_utils.FrameSequenceWithGroundTruth[RectifiedStereoFrame]:
        load_seed = self.base_seed if seed is None else seed
        resolved_root = None
        if self.data_root is not None:
            resolved_root = Path(self.data_root)
            if not resolved_root.is_absolute():
                resolved_root = Path(__file__).resolve().parents[2] / resolved_root

        return test_utils.load_euroc_sequence_segment(
            data_root=resolved_root,
            seq_name=self.seq_name,
            sequence_length=self.sequence_length,
            seed=load_seed,
            sampling_mode=self.sampling_mode,
            min_stride=self.min_stride,
            max_stride=self.max_stride,
            add_imu_noise=self.add_imu_noise,
            imu_noise_seed=self.imu_noise_seed,
            imu_noise_tau_bias=self.imu_noise_tau_bias,
            alpha=self.alpha,
            resize_to=self.resize_to,
        )

    def dataset_label(self) -> str:
        return self.seq_name
