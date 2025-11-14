"""Bundle adjustment utilities for the local VO pipeline."""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

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
        self._label = f" ({log_prefix})" if log_prefix else ""
        self._times: dict[str, float] = {}

    @contextmanager
    def section(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._times[name] = self._times.get(name, 0.0) + duration

    def report(self) -> None:
        if not self._times:
            return
        print(f"Bundle adjustment timing breakdown{self._label}:")
        longest = max(len(name) for name in self._times)
        for name, duration in sorted(self._times.items(), key=lambda item: item[1], reverse=True):
            print(f"  {name.ljust(longest)} : {duration:6.3f} s")


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

    def optimize(
        self,
        *,
        rectified_frames: Sequence[RectifiedStereoFrame],
        track_history: Sequence[Mapping[int, TrackObservation]],
        tracks: Mapping[int, FeatureTrack],
        sequence_results: Sequence[dict[str, Any]],
        sequence_sample: Any,
        use_imu: bool = True,
        gravity_vector: Sequence[float] | np.ndarray | None = None,
        log_prefix: str | None = None,
    ) -> dict[str, Any]:
        if not rectified_frames:
            raise ValueError("No frames provided for bundle adjustment.")

        profiler = _SectionProfiler(log_prefix)

        with profiler.section("select frames"):
            frames_for_ba = self._select_frames_for_ba(track_history)
        if not frames_for_ba:
            raise ValueError("No frames satisfy the observation thresholds for bundle adjustment.")

        with profiler.section("build calibrations"):
            calibration = self._build_calibration(rectified_frames[0])
            stereo_calibration = self._build_stereo_calibration(rectified_frames[0])

        with profiler.section("noise model setup"):
            measurement_noise, stereo_noise = self._build_measurement_noises()
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                self._as_array(self.config.pose_prior_sigmas)
            )

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        with profiler.section("pose initialization"):
            first_pose = sequence_sample.world_poses[0]
            gt_world_to_first = gtsam.Pose3(first_pose.rotation(), np.zeros(3))
            pose_initials = self._compute_pose_initializations(
                gt_world_to_first,
                sequence_results,
                len(rectified_frames),
            )

            for frame_idx in frames_for_ba:
                values.insert(X(frame_idx), self._clone_pose(pose_initials[frame_idx]))

        imu_available = bool(use_imu and getattr(sequence_sample, "imu_measurements", None))
        imu_bias_key = None
        imu_bias_initial = gtsam.imuBias.ConstantBias()
        bias_before_vec: np.ndarray | None = None
        bias_after_vec: np.ndarray | None = None
        imu_params = None

        if imu_available:
            with profiler.section("imu initialization"):
                imu_bias_key = B(frames_for_ba[0])
                imu_params = self._create_preintegration_params(gravity_vector)
                if not values.exists(imu_bias_key):
                    values.insert(imu_bias_key, imu_bias_initial)

                # Initialize velocities by differentiating consecutive poses.
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
                        imu_bias_initial,
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

        graph.add(gtsam.PriorFactorPose3(X(frames_for_ba[0]), gt_world_to_first, prior_noise))

        if imu_available and imu_bias_key is not None and values.exists(imu_bias_key):
            bias_before_vec = self._imu_bias_to_numpy(values.atConstantBias(imu_bias_key))

        with profiler.section("build landmark factors"):
            (
                landmark_keys,
                landmark_index_lookup,
                initial_landmarks,
                observations,
                stereo_counts,
                mono_counts,
            ) = self._build_landmark_factors(
                graph,
                values,
                rectified_frames,
                tracks,
                sequence_results,
                frames_for_ba,
                calibration,
                stereo_calibration,
                measurement_noise,
                stereo_noise,
                profiler,
            )

        if not observations:
            raise RuntimeError("No landmark observations available for bundle adjustment.")

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
                for key in range(len(landmark_keys))
            ]
            initial_pose_dict = {idx: values.atPose3(X(idx)) for idx in frames_for_ba}

            observation_matrix = [
                (frame_idx, landmark_idx, measurement)
                for frame_idx, landmark_idx, measurement in observations
            ]

        profiler.report()

        return {
            "frames_for_ba": frames_for_ba,
            "landmark_original_indices": landmark_index_lookup,
            "initial_pose_dict": initial_pose_dict,
            "optimized_pose_dict": optimized_pose_dict,
            "initial_landmarks": initial_landmarks,
            "optimized_landmarks": optimized_landmarks,
            "observation_matrix": observation_matrix,
            "stereo_counts": stereo_counts,
            "mono_counts": mono_counts,
            "imu_bias_before": bias_before_vec,
            "imu_bias_after": bias_after_vec,
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _select_frames_for_ba(
        self,
        track_history: Sequence[Mapping[int, TrackObservation]],
    ) -> list[int]:
        selected: list[int] = []
        for idx, observations in enumerate(track_history):
            if len(observations) >= self.config.min_observations_per_frame or idx == 0:
                selected.append(idx)
        if 0 not in selected:
            selected.insert(0, 0)
        return selected

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

    def _build_landmark_factors(
        self,
        graph: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
        rectified_frames: Sequence[RectifiedStereoFrame],
        tracks: Mapping[int, FeatureTrack],
        sequence_results: Sequence[dict[str, Any]],
        frames_for_ba: Sequence[int],
        calibration: gtsam.Cal3_S2,
        stereo_calibration: gtsam.Cal3_S2Stereo,
        measurement_noise: gtsam.noiseModel.Base,
        stereo_noise: gtsam.noiseModel.Base,
        profiler: _SectionProfiler | None = None,
    ) -> tuple[list[int], list[int], list[np.ndarray], list[tuple[int, int, np.ndarray]], dict[int, int], dict[int, int]]:
        landmark_keys: list[int] = []
        landmark_index_lookup: list[int] = []
        initial_landmarks: list[np.ndarray] = []
        observations: list[tuple[int, int, np.ndarray]] = []
        stereo_counts = {frame_idx: 0 for frame_idx in frames_for_ba}
        mono_counts = {frame_idx: 0 for frame_idx in frames_for_ba}
        landmark_limit = int(self.config.max_landmarks)
        if landmark_limit < 0:
            landmark_limit = 0
        frames_for_ba_set = set(frames_for_ba)
        camera_cache = self._build_camera_cache(frames_for_ba, values, rectified_frames)
        intrinsics = (
            float(calibration.fx()),
            float(calibration.fy()),
            float(calibration.px()),
            float(calibration.py()),
        )
        gating_threshold_sq = float(self.config.reprojection_gating_threshold_px) ** 2

        inlier_lookup = self._build_inlier_lookup(sequence_results)

        for track_id in sorted(tracks.keys()):
            if landmark_limit > 0 and len(landmark_keys) >= landmark_limit:
                break
            track = tracks[track_id]
            if not np.isfinite(track.anchor_depth) or track.anchor_depth <= 0.0:
                continue
            if track.anchor_frame not in frames_for_ba_set:
                continue

            with (profiler.section("world point init") if profiler else nullcontext()):
                T_B_from_S0 = rectified_frames[track.anchor_frame].calibration.T_B_from_S0
                anchor_pose = values.atPose3(X(track.anchor_frame))
                anchor_point_S0 = gtsam.Point3(*track.anchor_point3.tolist())
                world_point = anchor_pose.compose(T_B_from_S0).transformFrom(anchor_point_S0)
                world_point_np = self._point3_like_to_numpy(world_point)

            with (profiler.section("collect track observations") if profiler else nullcontext()):
                observation_frames = self._collect_track_observations(
                    track,
                    frames_for_ba_set,
                    camera_cache,
                    world_point_np,
                    track.anchor_frame,
                    inlier_lookup,
                    intrinsics,
                    gating_threshold_sq,
                )

            if len(observation_frames) < self.config.min_observations_per_landmark:
                continue

            with (profiler.section("factor creation") if profiler else nullcontext()):
                landmark_key = len(landmark_keys)
                landmark_keys.append(landmark_key)
                landmark_index_lookup.append(track_id)
                values.insert(L(landmark_key), world_point)
                initial_landmarks.append(world_point_np)

                for frame_idx, observation in observation_frames:
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
                        graph.add(
                            gtsam.GenericStereoFactor3D(
                                stereo_measurement,
                                stereo_noise,
                                X(frame_idx),
                                L(landmark_key),
                                stereo_calibration,
                                rectified_frames[frame_idx].calibration.T_B_from_S0,
                            )
                        )
                        stereo_counts[frame_idx] += 1
                    else:
                        depth_value = float(observation.depth)
                        if np.isfinite(depth_value) and depth_value > 0.0:
                            disparity = (stereo_calibration.fx() * stereo_calibration.baseline()) / depth_value
                            stereo_measurement = gtsam.StereoPoint2(
                                float(observation.keypoint[0]),
                                float(observation.keypoint[0] - disparity),
                                float(observation.keypoint[1]),
                            )
                            graph.add(
                                gtsam.GenericStereoFactor3D(
                                    stereo_measurement,
                                    stereo_noise,
                                    X(frame_idx),
                                    L(landmark_key),
                                    stereo_calibration,
                                    rectified_frames[frame_idx].calibration.T_B_from_S0,
                                )
                            )
                            stereo_counts[frame_idx] += 1
                        else:
                            graph.add(
                                gtsam.GenericProjectionFactorCal3_S2(
                                    gtsam.Point2(float(measurement_vec[0]), float(measurement_vec[1])),
                                    measurement_noise,
                                    X(frame_idx),
                                    L(landmark_key),
                                    calibration,
                                    rectified_frames[frame_idx].calibration.T_B_from_S0,
                                )
                            )
                            mono_counts[frame_idx] += 1

                    observations.append((frame_idx, landmark_key, measurement_vec))

        return (
            landmark_keys,
            landmark_index_lookup,
            initial_landmarks,
            observations,
            stereo_counts,
            mono_counts,
        )

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
        track: FeatureTrack,
        frames_for_ba_set: set[int],
        camera_cache: _CachedCameraBatch,
        world_point: np.ndarray,
        anchor_frame: int,
        inlier_lookup: dict[int, set[int]],
        intrinsics: tuple[float, float, float, float],
        gating_threshold_sq: float,
    ) -> list[tuple[int, TrackObservation]]:
        observation_frames: list[tuple[int, TrackObservation]] = []
        track_id = track.track_id
        eligible_frames: list[int] = []
        eligible_observations: list[TrackObservation] = []
        use_inliers = self.config.use_inlier_observations_only
        for frame_idx in track.observation_frames:
            if frame_idx not in frames_for_ba_set:
                continue
            observation = track.observations.get(frame_idx)
            if observation is None:
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
        max_observations = self.config.max_observations_per_landmark
        if max_observations > 0:
            max_observations = max(max_observations, self.config.min_observations_per_landmark)
        if max_observations > 0 and len(observation_frames) > max_observations:
            anchor_idx = None
            for idx, (frame_idx, _) in enumerate(observation_frames):
                if frame_idx == anchor_frame:
                    anchor_idx = idx
                    break
            if anchor_idx is None:
                anchor_idx = 0
            selected = [observation_frames[anchor_idx]]
            remaining = max_observations - 1
            if remaining > 0:
                indices = [i for i in range(len(observation_frames)) if i != anchor_idx]
                if remaining >= len(indices):
                    selected.extend(observation_frames[i] for i in indices)
                else:
                    step = len(indices) / remaining
                    idx_float = 0.0
                    for _ in range(remaining):
                        chosen = indices[int(idx_float)]
                        selected.append(observation_frames[chosen])
                        idx_float += step
            observation_frames = selected[:max_observations]

        observation_frames.sort(key=lambda item: item[0])

        return observation_frames

    @staticmethod
    def _build_inlier_lookup(sequence_results: Sequence[dict[str, Any]]) -> dict[int, set[int]]:
        lookup: dict[int, set[int]] = {}
        for result in sequence_results:
            if result.get("status") != "success":
                continue
            frame_idx = result["frame_index"]
            track_ids = result.get("track_ids")
            if track_ids is None:
                continue
            matches = result["matched_pair"].matches
            if hasattr(matches, "cpu"):
                matches = matches.cpu().numpy()
            if matches.size == 0:
                continue
            matched_track_ids = np.asarray(track_ids, dtype=int)
            valid_ids = matched_track_ids[matches[:, 0]]
            lookup.setdefault(frame_idx, set()).update(valid_ids.tolist())
        return lookup

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
