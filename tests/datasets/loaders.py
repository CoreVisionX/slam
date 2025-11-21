from __future__ import annotations

from dataclasses import dataclass

from registration.registration import RectifiedStereoFrame
import tests.test_utils as test_utils


@dataclass
class EurocSequenceLoader:
    """Loads a EuRoC V1/V2 sequence segment."""

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
        return test_utils.load_euroc_sequence_segment(
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


@dataclass
class TartanAirSequenceLoader:
    """Loads a TartanAir stereo sequence."""

    env: str = "AbandonedFactory"
    difficulty: str = "easy"
    traj: str = "P001"
    sequence_length: int = 6000
    base_seed: int = 0
    sampling_mode: str = "stride"
    min_stride: int = 1
    max_stride: int = 1
    load_ground_truth_depth: bool = False
    add_imu_noise: bool = True
    imu_noise_seed: int = 0
    imu_noise_tau_bias: float = 500.0

    def load_sequence(
        self,
        seed: int | None = None,
    ) -> test_utils.FrameSequenceWithGroundTruth[StereoFrame]:
        load_seed = self.base_seed if seed is None else seed
        return test_utils.load_tartanair_sequence_segment(
            env=self.env,
            difficulty=self.difficulty,
            traj=self.traj,
            sequence_length=self.sequence_length,
            seed=load_seed,
            sampling_mode=self.sampling_mode,
            min_stride=self.min_stride,
            max_stride=self.max_stride,
            load_ground_truth_depth=self.load_ground_truth_depth,
            add_imu_noise=self.add_imu_noise,
            imu_noise_seed=self.imu_noise_seed,
            imu_noise_tau_bias=self.imu_noise_tau_bias,
        )

    def dataset_label(self) -> str:
        return f"{self.env}_{self.traj}"
