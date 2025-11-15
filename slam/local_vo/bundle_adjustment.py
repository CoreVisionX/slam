"""Bundle adjustment utilities for the local VO pipeline."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, L, V, X

from registration.registration import RectifiedStereoFrame

from .klt_tracker import FeatureTrack, TrackObservation


@dataclass
class BundleAdjustmentConfig:
    """Configuration container for bundle adjustment."""

    min_observations_per_landmark: int = 3
    min_observations_per_frame: int = 5
    max_observations_per_landmark: int = 3
    max_landmarks: int = 10_000
    projection_noise_px: float = 1.0
    disparity_noise_px: float = 1.0
    use_huber_loss: bool = True
    reprojection_gating_threshold_px: float = 2.0
    use_inlier_observations_only: bool = False
    pose_prior_sigmas: Sequence[float] = field(
        default_factory=lambda: np.array(
            [
                np.deg2rad(1.0),
                np.deg2rad(1.0),
                np.deg2rad(1.0),
                0.10,
                0.10,
                0.10,
            ],
            dtype=float,
        )
        * 0.001,
    )
    imu_gravity_magnitude: float = 9.80665
    imu_accel_noise: float = 2.0e-3
    imu_gyro_noise: float = 1.7e-4
    imu_integration_noise: float = 5.0e-4
    imu_velocity_prior_sigma: float = 10.0
    imu_bias_prior_sigmas: Sequence[float] = field(
        default_factory=lambda: [
            3.0e-3,
            3.0e-3,
            3.0e-3,
            1.9393e-5,
            1.9393e-5,
            1.9393e-5,
        ]
    )
    imu_bias_prior_for_factors: Sequence[float] = field(
        default_factory=lambda: [0.01, 0.01, 0.01, 0.001, 0.001, 0.001]
    )


class _SectionProfiler:
    """Lightweight profiler that aggregates timing for named sections."""

    def __init__(self, log_prefix: str | None) -> None:
        self._log_prefix = log_prefix
        self._times: dict[str, float] = {}

    @contextmanager
    def section(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._times[name] = self._times.get(name, 0.0) + duration

    def _format_label(self, override: str | None) -> str:
        label = override if override is not None else self._log_prefix
        return f" ({label})" if label else ""

    def report(self, log_prefix: str | None = None) -> None:
        if not self._times:
            return
        label = self._format_label(log_prefix)
        print(f"Bundle adjustment timing breakdown{label}:")
        longest = max(len(name) for name in self._times)
        for name, duration in sorted(self._times.items(), key=lambda item: item[1], reverse=True):
            print(f"  {name.ljust(longest)} : {duration:6.3f} s")

    def report_average(self, call_count: int, log_prefix: str | None = None) -> None:
        if not self._times or call_count <= 0:
            return
        label = self._format_label(log_prefix)
        print(f"Average add_frame section times over {call_count} calls{label}:")
        longest = max(len(name) for name in self._times)
        for name, total_duration in sorted(self._times.items(), key=lambda item: item[1], reverse=True):
            avg_duration = total_duration / float(call_count)
            print(f"  {name.ljust(longest)} : {avg_duration:6.6f} s ({1/avg_duration:.2f} Hz)")


@dataclass(frozen=True)
class _CachedCameraBatch:
    frame_to_slot: dict[int, int]
    rotations_cw: np.ndarray
    translations_cw: np.ndarray

class IncrementalBundleAdjuster:
    """Simple iSAM-style bundle adjuster used by the experiments."""

    def __init__(self, config: BundleAdjustmentConfig | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise TypeError("Provide either a config object or keyword overrides, not both.")
        self.config = config or BundleAdjustmentConfig(**kwargs)
        self.reset()

    def reset(self, sequence_sample: Any | None = None) -> None:
        """Reset the incremental solver and optionally set the sequence context."""
        self._sequence_sample = sequence_sample
        self._gt_world_to_first: gtsam.Pose3 | None = None
        if sequence_sample is not None:
            first_pose = sequence_sample.world_poses[0]
            self._gt_world_to_first = gtsam.Pose3(first_pose.rotation(), np.zeros(3))
        self._frame_profiler = _SectionProfiler(None)
        self._add_frame_calls = 0
        self._vision_graph = gtsam.NonlinearFactorGraph()
        self._vision_values = gtsam.Values()
        self._frames_for_ba: list[int] = []
        self._frames_for_ba_set: set[int] = set()
        self._rectified_frame_buffer: list[RectifiedStereoFrame] = []
        self._track_history: list[Mapping[int, TrackObservation]] = []
        self._sequence_results: list[dict[str, Any]] = []
        self._inlier_lookup: dict[int, set[int]] = {}
        self._track_to_landmark: dict[int, int] = {}
        self._landmark_observed_frames: dict[int, set[int]] = {}
        self._landmark_index_lookup: list[int] = []
        self._initial_landmarks: list[np.ndarray] = []
        self._observation_matrix: list[tuple[int, int, np.ndarray]] = []
        self._stereo_counts: dict[int, int] = {}
        self._mono_counts: dict[int, int] = {}
        self._initial_pose_dict: dict[int, gtsam.Pose3] = {}
        self._last_pose_initial: gtsam.Pose3 | None = None
        self._camera_cache: _CachedCameraBatch | None = None
        self._calibration: gtsam.Cal3_S2 | None = None
        self._stereo_calibration: gtsam.Cal3_S2Stereo | None = None
        self._intrinsics: tuple[float, float, float, float] | None = None
        self._measurement_noise, self._stereo_noise = self._build_measurement_noises()
        self._prior_noise = gtsam.noiseModel.Diagonal.Sigmas(self._as_array(self.config.pose_prior_sigmas))
        self._gating_threshold_sq = float(self.config.reprojection_gating_threshold_px) ** 2
        self._track_landmark_progress: dict[int, int] = {}
        self._pending_landmark_observations: dict[int, list[tuple[int, TrackObservation]]] = {}

    def add_frame(
        self,
        *,
        rectified_frame: RectifiedStereoFrame,
        track_observations: Mapping[int, TrackObservation],
        tracks: Mapping[int, FeatureTrack],
        sequence_result: Mapping[str, Any] | None = None,
    ) -> None:
        """Incrementally add pose and landmark factors for each processed frame."""
        if self._sequence_sample is None:
            raise RuntimeError(
                "IncrementalBundleAdjuster.reset(sequence_sample=...) must be called before add_frame()."
            )
        if tracks is None:
            raise ValueError("tracks must be provided to add_frame for incremental updates.")

        frame_idx = len(self._rectified_frame_buffer)
        self._rectified_frame_buffer.append(rectified_frame)
        self._track_history.append(dict(track_observations))
        if sequence_result is not None:
            self._sequence_results.append(sequence_result)
            self._update_inlier_lookup(sequence_result)

        pose_initial = self._update_pose_initial(frame_idx, sequence_result)
        include_frame = self._should_include_frame(frame_idx, track_observations)

        if include_frame:
            with self._frame_profiler.section("pose initialization"):
                self._ensure_calibrations(rectified_frame)
                self._insert_pose_initial(frame_idx, pose_initial)

        with self._frame_profiler.section("landmark update"):
            if self._frames_for_ba:
                track_ids_for_update: Iterable[int] | None = None
                if track_observations:
                    track_ids_for_update = track_observations.keys()
                self._update_landmarks(tracks, track_ids_for_update)

        self._add_frame_calls += 1

    def optimize(
        self,
        *,
        sequence_sample: Any,
        tracks: Mapping[int, FeatureTrack] | None = None,  # retained for API compatibility
        use_imu: bool = True,
        gravity_vector: Sequence[float] | np.ndarray | None = None,
        log_prefix: str | None = None,
    ) -> dict[str, Any]:
        if self._sequence_sample is None:
            self._sequence_sample = sequence_sample
        elif sequence_sample is not self._sequence_sample:
            self._sequence_sample = sequence_sample

        if not self._frames_for_ba:
            raise ValueError("No frames have been added to the bundle adjuster.")
        if not self._track_to_landmark:
            raise RuntimeError("No landmark observations available for bundle adjustment.")

        self._frame_profiler.report_average(self._add_frame_calls, log_prefix)
        profiler = _SectionProfiler(log_prefix)
        frames_for_ba = list(self._frames_for_ba)
        graph = gtsam.NonlinearFactorGraph(self._vision_graph)
        values = gtsam.Values(self._vision_values)

        sequence_sample = self._sequence_sample
        imu_available = bool(use_imu and getattr(sequence_sample, "imu_measurements", None))
        imu_bias_key = None
        bias_before_vec: np.ndarray | None = None
        bias_after_vec: np.ndarray | None = None
        imu_params = None

        if imu_available:
            with profiler.section("imu initialization"):
                imu_bias_key = B(frames_for_ba[0])
                imu_params = self._create_preintegration_params(gravity_vector)
                if not values.exists(imu_bias_key):
                    values.insert(imu_bias_key, gtsam.imuBias.ConstantBias())

                if len(frames_for_ba) >= 2:
                    first_velocity = self._finite_difference_velocity(
                        frames_for_ba[0],
                        frames_for_ba[1],
                        values,
                        sequence_sample.frame_timestamps,
                    )
                    values.insert(V(frames_for_ba[0]), first_velocity)

                    for prev_frame, next_frame in zip(frames_for_ba[:-1], frames_for_ba[1:]):
                        velocity = self._finite_difference_velocity(
                            prev_frame,
                            next_frame,
                            values,
                            sequence_sample.frame_timestamps,
                        )
                        values.insert(V(next_frame), velocity)

                bias_prior = gtsam.noiseModel.Diagonal.Sigmas(
                    self._as_array(self.config.imu_bias_prior_for_factors)
                )
                graph.add(
                    gtsam.PriorFactorConstantBias(
                        B(frames_for_ba[0]),
                        gtsam.imuBias.ConstantBias(),
                        bias_prior,
                    )
                )

                for frame_idx in frames_for_ba:
                    if not values.exists(B(frame_idx)):
                        values.insert(B(frame_idx), gtsam.imuBias.ConstantBias())

            with profiler.section("add imu factors"):
                self._add_imu_factors(
                    graph,
                    values,
                    imu_params,
                    sequence_sample,
                    frames_for_ba,
                )

        if imu_available and imu_bias_key is not None and values.exists(imu_bias_key):
            bias_before_vec = self._imu_bias_to_numpy(values.atConstantBias(imu_bias_key))

        label = f" ({log_prefix})" if log_prefix else ""
        print(f"Performing bundle adjustment{label}...")
        with profiler.section("optimizer"):
            start_time = time.perf_counter()
            optimized_values = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
            end_time = time.perf_counter()
        print(f"Bundle adjustment optimized in {end_time - start_time:.2f} seconds{label}")

        if imu_available:
            self._print_imu_diagnostics(graph, values)

        if imu_available and imu_bias_key is not None and optimized_values.exists(imu_bias_key):
            bias_after_vec = self._imu_bias_to_numpy(optimized_values.atConstantBias(imu_bias_key))

        with profiler.section("result packaging"):
            optimized_pose_dict = {idx: optimized_values.atPose3(X(idx)) for idx in frames_for_ba}
            optimized_landmarks = [
                self._point3_like_to_numpy(optimized_values.atPoint3(L(key)))
                for key in range(len(self._landmark_index_lookup))
            ]

        profiler.report()

        return {
            "frames_for_ba": frames_for_ba,
            "landmark_original_indices": list(self._landmark_index_lookup),
            "initial_pose_dict": dict(self._initial_pose_dict),
            "optimized_pose_dict": optimized_pose_dict,
            "initial_landmarks": list(self._initial_landmarks),
            "optimized_landmarks": optimized_landmarks,
            "observation_matrix": list(self._observation_matrix),
            "stereo_counts": dict(self._stereo_counts),
            "mono_counts": dict(self._mono_counts),
            "imu_bias_before": bias_before_vec,
            "imu_bias_after": bias_after_vec,
        }

    @staticmethod
    def _build_calibration(frame: RectifiedStereoFrame) -> gtsam.Cal3_S2:
        K = frame.calibration.K_left_rect
        return gtsam.Cal3_S2(
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
        )

    @staticmethod
    def _build_stereo_calibration(frame: RectifiedStereoFrame) -> gtsam.Cal3_S2Stereo:
        calib = frame.calibration
        K = calib.K_left_rect
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        skew = float(K[0, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        baseline = float(np.abs(calib.P_right[0, 3] / calib.P_right[0, 0]))
        if baseline <= 0.0:
            baseline = float(np.linalg.norm(calib.T.reshape(-1)))
        return gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, baseline)

    def _build_measurement_noises(self) -> tuple[gtsam.noiseModel.Base, gtsam.noiseModel.Base]:
        if self.config.use_huber_loss:
            measurement_noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber(1.345),
                gtsam.noiseModel.Isotropic.Sigma(2, self.config.projection_noise_px),
            )
            stereo_noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber(1.345),
                gtsam.noiseModel.Diagonal.Sigmas(
                    np.array(
                        [
                            self.config.projection_noise_px,
                            self.config.projection_noise_px,
                            self.config.disparity_noise_px,
                        ],
                        dtype=float,
                    )
                ),
            )
        else:
            measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, self.config.projection_noise_px)
            stereo_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array(
                    [
                        self.config.projection_noise_px,
                        self.config.projection_noise_px,
                        self.config.projection_noise_px,
                    ],
                    dtype=float,
                )
            )
        return measurement_noise, stereo_noise

    @staticmethod
    def _compute_pose_initializations(
        rotate_world_to_first: gtsam.Pose3,
        sequence_results: Sequence[dict[str, Any]],
        sequence_length: int,
    ) -> list[gtsam.Pose3]:
        initializations: list[gtsam.Pose3] = [rotate_world_to_first]
        result_by_frame = {
            result["frame_index"]: result
            for result in sequence_results
            if result.get("status") == "success"
        }
        current_pose = rotate_world_to_first
        for frame_idx in range(1, sequence_length):
            if frame_idx in result_by_frame:
                current_pose = result_by_frame[frame_idx]["estimated_pose"]
            initializations.append(current_pose)
        return initializations

    @staticmethod
    def _clone_pose(pose: gtsam.Pose3) -> gtsam.Pose3:
        return gtsam.Pose3(pose.rotation(), pose.translation())

    @staticmethod
    def _point3_like_to_numpy(point: Any) -> np.ndarray:
        if hasattr(point, "x"):
            return np.array([float(point.x()), float(point.y()), float(point.z())], dtype=np.float64)
        return np.asarray(point, dtype=np.float64)

    @staticmethod
    def _imu_bias_to_numpy(bias: gtsam.imuBias.ConstantBias) -> np.ndarray:
        accel = bias.accelerometer()
        gyro = bias.gyroscope()
        return np.array(
            [
                float(accel[0]),
                float(accel[1]),
                float(accel[2]),
                float(gyro[0]),
                float(gyro[1]),
                float(gyro[2]),
            ],
            dtype=np.float64,
        )

    def _create_preintegration_params(
        self,
        gravity_vector: Sequence[float] | np.ndarray | None,
    ) -> gtsam.PreintegrationCombinedParams:
        params = gtsam.PreintegrationCombinedParams.MakeSharedU(self.config.imu_gravity_magnitude)
        if gravity_vector is not None:
            params.n_gravity = np.asarray(gravity_vector, dtype=float)
        else:
            params.n_gravity = np.array([0.0, 0.0, -self.config.imu_gravity_magnitude], dtype=float)
        params.setAccelerometerCovariance(np.eye(3) * (self.config.imu_accel_noise**2))
        params.setGyroscopeCovariance(np.eye(3) * (self.config.imu_gyro_noise**2))
        params.setIntegrationCovariance(np.eye(3) * (self.config.imu_integration_noise**2))
        params.setBiasAccCovariance(
            np.eye(3) * (self._as_array(self.config.imu_bias_prior_sigmas)[0] ** 2)
        )
        params.setBiasOmegaCovariance(
            np.eye(3) * (self._as_array(self.config.imu_bias_prior_sigmas)[3] ** 2)
        )
        return params

    def _add_imu_factors(
        self,
        graph: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
        imu_params: gtsam.PreintegrationCombinedParams,
        sequence_sample: Any,
        frames_for_ba: Sequence[int],
    ) -> None:
        for prev_frame, next_frame in zip(frames_for_ba[:-1], frames_for_ba[1:]):
            pim = gtsam.PreintegratedCombinedMeasurements(
                imu_params,
                values.atConstantBias(B(prev_frame)),
            )
            batch = sequence_sample.imu_measurements[next_frame]
            self._integrate_imu_batch(pim, batch)

            expected_dt = sequence_sample.frame_timestamps[next_frame] - sequence_sample.frame_timestamps[prev_frame]
            if not np.isclose(pim.deltaTij(), expected_dt, atol=1e-4):
                raise ValueError(
                    f"IMU Δt {pim.deltaTij():.6f} vs expected {expected_dt:.6f} between {prev_frame} and {next_frame}"
                )

            graph.add(
                gtsam.CombinedImuFactor(
                    X(prev_frame),
                    V(prev_frame),
                    X(next_frame),
                    V(next_frame),
                    B(prev_frame),
                    B(next_frame),
                    pim,
                )
            )

    @staticmethod
    def _integrate_imu_batch(
        pim: gtsam.PreintegratedImuMeasurements,
        batch: Any,
    ) -> None:
        if batch is None:
            raise ValueError("IMU batch missing while attempting to integrate measurements.")
        for i in range(len(batch)):
            pim.integrateMeasurement(
                batch.linear_accelerations[i],
                batch.angular_velocities[i],
                deltaT=batch.dts[i],
            )

    @staticmethod
    def _finite_difference_velocity(
        prev_frame: int,
        next_frame: int,
        values: gtsam.Values,
        frame_timestamps: Sequence[float],
    ) -> gtsam.Point3:
        prev_pose = values.atPose3(X(prev_frame))
        next_pose = values.atPose3(X(next_frame))
        dt = frame_timestamps[next_frame] - frame_timestamps[prev_frame]
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("Invalid timestamp delta for velocity initialization.")
        return (next_pose.translation() - prev_pose.translation()) / dt

    def _should_include_frame(
        self,
        frame_idx: int,
        track_observations: Mapping[int, TrackObservation],
    ) -> bool:
        return frame_idx == 0 or len(track_observations) >= self.config.min_observations_per_frame

    def _ensure_calibrations(self, frame: RectifiedStereoFrame) -> None:
        if self._calibration is None:
            self._calibration = self._build_calibration(frame)
            self._stereo_calibration = self._build_stereo_calibration(frame)
            self._intrinsics = (
                float(self._calibration.fx()),
                float(self._calibration.fy()),
                float(self._calibration.px()),
                float(self._calibration.py()),
            )

    def _update_pose_initial(
        self,
        frame_idx: int,
        sequence_result: Mapping[str, Any] | None,
    ) -> gtsam.Pose3:
        if frame_idx == 0:
            if self._gt_world_to_first is None:
                if self._sequence_sample is None:
                    raise RuntimeError("Sequence sample missing for pose initialization.")
                first_pose = self._sequence_sample.world_poses[0]
                self._gt_world_to_first = gtsam.Pose3(first_pose.rotation(), np.zeros(3))
            pose_guess = self._gt_world_to_first
        else:
            pose_guess = self._last_pose_initial or self._gt_world_to_first
            if pose_guess is None:
                raise RuntimeError("Initial pose unavailable; ensure reset() was called with a sequence sample.")
        if sequence_result and sequence_result.get("status") == "success":
            pose_guess = sequence_result["estimated_pose"]
        self._last_pose_initial = pose_guess
        return pose_guess

    def _insert_pose_initial(self, frame_idx: int, pose_initial: gtsam.Pose3) -> None:
        if frame_idx in self._initial_pose_dict:
            return
        if self._calibration is None or self._stereo_calibration is None:
            raise RuntimeError("Camera calibration must be initialized before inserting poses.")
        if not self._frames_for_ba:
            if self._gt_world_to_first is None:
                raise RuntimeError("Ground truth orientation for first frame missing.")
            self._vision_graph.add(
                gtsam.PriorFactorPose3(X(frame_idx), self._gt_world_to_first, self._prior_noise)
            )

        self._vision_values.insert(X(frame_idx), self._clone_pose(pose_initial))
        self._initial_pose_dict[frame_idx] = self._clone_pose(pose_initial)
        self._frames_for_ba.append(frame_idx)
        self._frames_for_ba_set.add(frame_idx)
        self._stereo_counts.setdefault(frame_idx, 0)
        self._mono_counts.setdefault(frame_idx, 0)
        self._camera_cache = None

    def _update_inlier_lookup(self, sequence_result: Mapping[str, Any]) -> None:
        if sequence_result.get("status") != "success":
            return
        track_ids = sequence_result.get("track_ids")
        matched_pair = sequence_result.get("matched_pair")
        if track_ids is None or matched_pair is None:
            return
        matches = matched_pair.matches
        if hasattr(matches, "cpu"):
            matches = matches.cpu().numpy()
        if matches.size == 0:
            return
        matched_track_ids = np.asarray(track_ids, dtype=int)
        valid_ids = matched_track_ids[matches[:, 0]]
        frame_idx = sequence_result["frame_index"]
        lookup = self._inlier_lookup.setdefault(frame_idx, set())
        lookup.update(valid_ids.tolist())

    def _get_camera_cache(self) -> _CachedCameraBatch:
        if self._camera_cache is None:
            self._camera_cache = self._build_camera_cache(
                self._frames_for_ba,
                self._vision_values,
                self._rectified_frame_buffer,
            )
        return self._camera_cache

    def _landmark_limit_reached(self) -> bool:
        limit = int(self.config.max_landmarks)
        return limit > 0 and len(self._landmark_index_lookup) >= limit

    def _update_landmarks(
        self,
        tracks: Mapping[int, FeatureTrack],
        track_ids_to_update: Iterable[int] | None = None,
    ) -> None:
        if not tracks or not self._frames_for_ba:
            return
        if self._calibration is None or self._stereo_calibration is None or self._intrinsics is None:
            return
        profiler = self._frame_profiler
        with profiler.section("landmark track selection"):
            if track_ids_to_update is not None:
                unique_ids: list[int] = []
                seen: set[int] = set()
                for track_id in track_ids_to_update:
                    if track_id in seen:
                        continue
                    if track_id not in tracks:
                        continue
                    unique_ids.append(track_id)
                    seen.add(track_id)
                if not unique_ids:
                    return
                track_iter: Iterable[FeatureTrack] = (tracks[track_id] for track_id in unique_ids)
            else:
                track_iter = tracks.values()
        with profiler.section("landmark camera cache"):
            frames_for_ba_set = self._frames_for_ba_set
            camera_cache = self._get_camera_cache()
            intrinsics = self._intrinsics
        min_obs = int(self.config.min_observations_per_landmark)
        max_obs = int(self.config.max_observations_per_landmark)
        if max_obs > 0:
            max_obs = max(max_obs, min_obs)

        for track in track_iter:
            track_id = track.track_id
            has_landmark = track_id in self._track_to_landmark
            if not has_landmark and self._landmark_limit_reached():
                continue
            if track.anchor_frame not in frames_for_ba_set:
                continue
            if not np.isfinite(track.anchor_depth) or track.anchor_depth <= 0.0:
                continue
            if track.anchor_frame >= len(self._rectified_frame_buffer):
                continue

            with profiler.section("landmark world point"):
                anchor_pose = self._vision_values.atPose3(X(track.anchor_frame))
                T_B_from_S0 = self._rectified_frame_buffer[track.anchor_frame].calibration.T_B_from_S0
                anchor_point_S0 = gtsam.Point3(*track.anchor_point3.tolist())
                world_point = anchor_pose.compose(T_B_from_S0).transformFrom(anchor_point_S0)
                world_point_np = self._point3_like_to_numpy(world_point)

            observations_list = track.observation_frames
            processed_count = self._track_landmark_progress.get(track_id, 0)
            if processed_count >= len(observations_list):
                continue
            candidate_frames = observations_list[processed_count:]
            candidate_pairs: list[tuple[int, TrackObservation]] = []
            for frame_idx in candidate_frames:
                observation = track.observations.get(frame_idx)
                if observation is None:
                    continue
                candidate_pairs.append((frame_idx, observation))
            if not candidate_pairs:
                self._track_landmark_progress[track_id] = len(observations_list)
                continue
            candidate_frames = [item[0] for item in candidate_pairs]
            candidate_observations = [item[1] for item in candidate_pairs]
            self._track_landmark_progress[track_id] = len(observations_list)

            with profiler.section("landmark collect observations"):
                observation_frames = self._collect_track_observations(
                    track_id,
                    candidate_frames,
                    candidate_observations,
                    frames_for_ba_set,
                    camera_cache,
                    world_point_np,
                    track.anchor_frame,
                    self._inlier_lookup,
                    intrinsics,
                    self._gating_threshold_sq,
                )
            if has_landmark:
                if not observation_frames:
                    continue
                if max_obs > 0:
                    observed_frames = self._landmark_observed_frames.get(track_id, set())
                    remaining = max(max_obs - len(observed_frames), 0)
                    if remaining <= 0:
                        continue
                    if len(observation_frames) > remaining:
                        observation_frames = self._limit_observation_frames(
                            observation_frames, track.anchor_frame, remaining
                        )
                        if not observation_frames:
                            continue
                with profiler.section("landmark add observations"):
                    self._add_landmark_observations(track, world_point, world_point_np, observation_frames)
                continue

            pending = self._pending_landmark_observations.setdefault(track_id, [])
            if observation_frames:
                pending.extend(observation_frames)
            if len(pending) < min_obs:
                continue
            pending.sort(key=lambda item: item[0])
            candidate = pending
            if max_obs > 0 and len(candidate) > max_obs:
                candidate = self._limit_observation_frames(candidate, track.anchor_frame, max_obs)
            with profiler.section("landmark add observations"):
                self._add_landmark_observations(track, world_point, world_point_np, candidate)
            self._pending_landmark_observations.pop(track_id, None)

    def _add_landmark_observations(
        self,
        track: FeatureTrack,
        world_point: gtsam.Point3,
        world_point_np: np.ndarray,
        observation_frames: list[tuple[int, TrackObservation]],
    ) -> None:
        stereo_calibration = self._stereo_calibration
        calibration = self._calibration
        if stereo_calibration is None or calibration is None:
            raise RuntimeError("Calibrations must be initialized before adding landmarks.")

        track_id = track.track_id
        if track_id not in self._track_to_landmark:
            landmark_key = len(self._track_to_landmark)
            self._track_to_landmark[track_id] = landmark_key
            self._landmark_index_lookup.append(track_id)
            self._vision_values.insert(L(landmark_key), world_point)
            self._initial_landmarks.append(world_point_np)
            self._landmark_observed_frames[track_id] = set()
        else:
            landmark_key = self._track_to_landmark[track_id]

        observed_frames = self._landmark_observed_frames[track_id]
        for frame_idx, observation in observation_frames:
            if frame_idx in observed_frames:
                continue
            measurement_vec = np.asarray(observation.keypoint, dtype=np.float64)
            if frame_idx == track.anchor_frame:
                depth_value = float(track.anchor_depth)
                if not np.isfinite(depth_value) or depth_value <= 0.0:
                    continue
                disparity = (stereo_calibration.fx() * stereo_calibration.baseline()) / depth_value
                stereo_measurement = gtsam.StereoPoint2(
                    float(track.anchor_keypoint[0]),
                    float(track.anchor_keypoint[0] - disparity),
                    float(track.anchor_keypoint[1]),
                )
                self._vision_graph.add(
                    gtsam.GenericStereoFactor3D(
                        stereo_measurement,
                        self._stereo_noise,
                        X(frame_idx),
                        L(landmark_key),
                        stereo_calibration,
                        self._rectified_frame_buffer[frame_idx].calibration.T_B_from_S0,
                    )
                )
                self._stereo_counts[frame_idx] += 1
            else:
                depth_value = float(observation.depth)
                if np.isfinite(depth_value) and depth_value > 0.0:
                    disparity = (stereo_calibration.fx() * stereo_calibration.baseline()) / depth_value
                    stereo_measurement = gtsam.StereoPoint2(
                        float(observation.keypoint[0]),
                        float(observation.keypoint[0] - disparity),
                        float(observation.keypoint[1]),
                    )
                    self._vision_graph.add(
                        gtsam.GenericStereoFactor3D(
                            stereo_measurement,
                            self._stereo_noise,
                            X(frame_idx),
                            L(landmark_key),
                            stereo_calibration,
                            self._rectified_frame_buffer[frame_idx].calibration.T_B_from_S0,
                        )
                    )
                    self._stereo_counts[frame_idx] += 1
                else:
                    self._vision_graph.add(
                        gtsam.GenericProjectionFactorCal3_S2(
                            gtsam.Point2(float(measurement_vec[0]), float(measurement_vec[1])),
                            self._measurement_noise,
                            X(frame_idx),
                            L(landmark_key),
                            calibration,
                            self._rectified_frame_buffer[frame_idx].calibration.T_B_from_S0,
                        )
                    )
                    self._mono_counts[frame_idx] += 1
            observed_frames.add(frame_idx)
            self._observation_matrix.append((frame_idx, landmark_key, measurement_vec))

    def _build_camera_cache(
        self,
        frames_for_ba: Sequence[int],
        values: gtsam.Values,
        rectified_frames: Sequence[RectifiedStereoFrame],
    ) -> _CachedCameraBatch:
        """Precompute world-to-camera rotations/translations for BA frames."""
        frame_to_slot: dict[int, int] = {}
        rotations: list[np.ndarray] = []
        translations: list[np.ndarray] = []
        for slot, frame_idx in enumerate(frames_for_ba):
            pose_initial = values.atPose3(X(frame_idx))
            T_B_from_S0 = rectified_frames[frame_idx].calibration.T_B_from_S0
            cam_pose = pose_initial.compose(T_B_from_S0)
            rot_world_from_cam = cam_pose.rotation().matrix()
            t_world_from_cam = self._point3_like_to_numpy(cam_pose.translation())
            rot_cam_from_world = rot_world_from_cam.T
            t_cam_from_world = -rot_cam_from_world @ t_world_from_cam
            frame_to_slot[frame_idx] = slot
            rotations.append(rot_cam_from_world.astype(np.float64, copy=False))
            translations.append(t_cam_from_world.astype(np.float64, copy=False))
        return _CachedCameraBatch(
            frame_to_slot=frame_to_slot,
            rotations_cw=np.asarray(rotations, dtype=np.float64),
            translations_cw=np.asarray(translations, dtype=np.float64),
        )

    @staticmethod
    def _project_point_fast(
        rotation_cw: np.ndarray,
        translation_cw: np.ndarray,
        world_point: np.ndarray,
        intrinsics: tuple[float, float, float, float],
    ) -> np.ndarray | None:
        """Project a 3D world point using cached camera extrinsics/intrinsics."""
        cam_point = rotation_cw @ world_point + translation_cw
        z = float(cam_point[2])
        if not np.isfinite(z) or z <= 0.0:
            return None
        inv_z = 1.0 / z
        fx, fy, px, py = intrinsics
        x = float(cam_point[0]) * inv_z
        y = float(cam_point[1]) * inv_z
        return np.array([fx * x + px, fy * y + py], dtype=np.float64)

    def _collect_track_observations(
        self,
        track_id: int,
        candidate_frames: Sequence[int],
        candidate_observations: Sequence[TrackObservation],
        frames_for_ba_set: set[int],
        camera_cache: _CachedCameraBatch,
        world_point: np.ndarray,
        anchor_frame: int,
        inlier_lookup: dict[int, set[int]],
        intrinsics: tuple[float, float, float, float],
        gating_threshold_sq: float,
    ) -> list[tuple[int, TrackObservation]]:
        observation_frames: list[tuple[int, TrackObservation]] = []
        eligible_frames: list[int] = []
        eligible_observations: list[TrackObservation] = []
        use_inliers = self.config.use_inlier_observations_only
        for frame_idx, observation in zip(candidate_frames, candidate_observations):
            if frame_idx not in frames_for_ba_set:
                continue
            if (
                use_inliers
                and frame_idx != anchor_frame
                and frame_idx in inlier_lookup
                and track_id not in inlier_lookup[frame_idx]
            ):
                continue
            if frame_idx not in camera_cache.frame_to_slot:
                continue
            eligible_frames.append(frame_idx)
            eligible_observations.append(observation)

        if not eligible_frames:
            return observation_frames

        if len(eligible_frames) <= 4:
            for frame_idx, observation in zip(eligible_frames, eligible_observations):
                slot = camera_cache.frame_to_slot[frame_idx]
                rotation_cw = camera_cache.rotations_cw[slot]
                translation_cw = camera_cache.translations_cw[slot]
                predicted_vec = self._project_point_fast(rotation_cw, translation_cw, world_point, intrinsics)
                if predicted_vec is None:
                    continue
                measurement_vec = observation.keypoint
                residual = predicted_vec - measurement_vec
                if float(residual[0] * residual[0] + residual[1] * residual[1]) > gating_threshold_sq:
                    continue
                observation_frames.append((frame_idx, observation))
            return observation_frames

        slot_indices = np.array([camera_cache.frame_to_slot[idx] for idx in eligible_frames], dtype=np.int64)
        rotations = camera_cache.rotations_cw[slot_indices]
        translations = camera_cache.translations_cw[slot_indices]
        cam_points = np.einsum("nij,j->ni", rotations, world_point) + translations
        z = cam_points[:, 2]
        valid_mask = np.isfinite(z) & (z > 0.0)
        inv_z = np.zeros_like(z)
        inv_z[valid_mask] = 1.0 / z[valid_mask]
        fx, fy, px, py = intrinsics
        predicted = np.empty((cam_points.shape[0], 2), dtype=np.float64)
        predicted[:, 0] = fx * cam_points[:, 0] * inv_z + px
        predicted[:, 1] = fy * cam_points[:, 1] * inv_z + py
        measurements = np.vstack([obs.keypoint for obs in eligible_observations]).astype(np.float64)
        residuals = predicted - measurements
        residual_sq = residuals[:, 0] * residuals[:, 0] + residuals[:, 1] * residuals[:, 1]
        passing = valid_mask & np.isfinite(residual_sq) & (residual_sq <= gating_threshold_sq)

        for idx, keep in enumerate(passing):
            if keep:
                observation_frames.append((eligible_frames[idx], eligible_observations[idx]))
        observation_frames.sort(key=lambda item: item[0])

        return observation_frames

    @staticmethod
    def _limit_observation_frames(
        observation_frames: Sequence[tuple[int, TrackObservation]],
        anchor_frame: int,
        max_observations: int,
    ) -> list[tuple[int, TrackObservation]]:
        if max_observations <= 0 or len(observation_frames) <= max_observations:
            return list(observation_frames)
        anchor_entry: tuple[int, TrackObservation] | None = None
        remaining_entries: list[tuple[int, TrackObservation]] = []
        for frame_idx, obs in observation_frames:
            if frame_idx == anchor_frame and anchor_entry is None:
                anchor_entry = (frame_idx, obs)
            else:
                remaining_entries.append((frame_idx, obs))
        selected: list[tuple[int, TrackObservation]] = []
        if anchor_entry is not None:
            selected.append(anchor_entry)
        available = max_observations - len(selected)
        if available <= 0:
            return selected
        if available >= len(remaining_entries):
            selected.extend(remaining_entries[:available])
            return selected[:max_observations]
        step = len(remaining_entries) / available
        idx_float = 0.0
        for _ in range(available):
            chosen_idx = int(idx_float)
            selected.append(remaining_entries[chosen_idx])
            idx_float += step
        return selected[:max_observations]

    @staticmethod
    def _print_imu_diagnostics(
        graph: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
    ) -> None:
        vision_err = 0.0
        imu_err = 0.0
        for idx in range(graph.size()):
            factor = graph.at(idx)
            name = type(factor).__name__
            if "ProjectionFactor" in name or "StereoFactor" in name:
                vision_err += float(factor.error(values))
            if "ImuFactor" in name:
                imu_err += float(factor.error(values))

        print(f"[diag] Vision error evaluated on IMU-only poses: {vision_err:.6f}")
        print(f"[diag] IMU error evaluated on IMU-only poses: {imu_err:.6f}")

    @staticmethod
    def _as_array(values: Sequence[float]) -> np.ndarray:
        return np.asarray(values, dtype=float)
