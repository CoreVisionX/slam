"""High-level orchestration for the stereo SLAM pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import gtsam
import numpy as np

from backend.pose_graph import GtsamPoseGraph
from backend.proximity_loop_detector import ProximityLoopDetector
from depth.sgbm import SGBM
from registration.registration import StereoDepthFrame, StereoFrame
from slam.frontend import FrontendOutput, FrontendTimings, StereoFrontend
from slam.logger import RerunLogger, TrajectoryMetrics
from slam.loop_closure import LoopClosureManager, LoopClosureResult
from slam.matcher_factory import MatcherType, create_matcher
from slam.metrics import PerformanceSnapshot, PerformanceTracker


@dataclass(slots=True)
class SlamConfig:
    keyframe_translation_threshold: float = 1.5
    keyframe_rotation_threshold: float = float(np.deg2rad(24.0))
    align_ground_truth: bool = True
    enable_rerun_logging: bool = True
    loop_min_inliers: int = 100
    use_huber_loss: bool = True
    huber_k: float = 1.0
    rerun_app_id: str = "slam"
    rerun_tcp_address: str | None = None
    proximity_max_translation: float = 12.0
    proximity_max_rotation: float = float(np.deg2rad(60.0))
    proximity_max_candidates: int = 20
    proximity_min_separation: int = 1
    loop_worker_count: int | None = None
    odometry_noise_sigmas: np.ndarray = field(
        default_factory=lambda: np.array(
            [np.deg2rad(0.3), np.deg2rad(0.3), np.deg2rad(0.3), 0.007, 0.007, 0.007],
            dtype=float,
        )
    )
    loop_noise_sigmas: np.ndarray = field(
        default_factory=lambda: np.array(
            [np.deg2rad(0.3), np.deg2rad(0.3), np.deg2rad(0.3), 0.014, 0.014, 0.014],
            dtype=float,
        )
    )
    absolute_rotation_noise_sigmas: np.ndarray | None = field(
        default_factory=lambda: np.array(
            [np.deg2rad(1.0), np.deg2rad(1.0), np.deg2rad(1.0)],
            dtype=float,
        )
    )
    feature_matcher: MatcherType = "lighterglue"
    rectify_inputs: bool = True


@dataclass(slots=True)
class SlamStepResult:
    frame_index: int
    is_keyframe: bool
    loop_closures_added: int
    latest_pose: gtsam.Pose3
    frontend_timings: FrontendTimings | None
    total_time: float
    metrics: TrajectoryMetrics | None
    performance: PerformanceSnapshot


class StereoSlamSystem:
    """User-facing façade wrapping the entire SLAM pipeline."""

    def __init__(
        self,
        config: SlamConfig,
        *,
        pose_graph: GtsamPoseGraph,
        frontend: StereoFrontend,
        loop_detector: ProximityLoopDetector,
        loop_manager: LoopClosureManager,
        logger: RerunLogger | None = None,
    ) -> None:
        self.config = config
        self.pose_graph = pose_graph
        self.frontend = frontend
        self.loop_detector = loop_detector
        self.loop_manager = loop_manager
        self.logger = logger
        self.performance = PerformanceTracker(history=300)

        self.loop_manager.start()

        self._odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(self.config.odometry_noise_sigmas)
        base_loop_noise = gtsam.noiseModel.Diagonal.Sigmas(self.config.loop_noise_sigmas)
        if self.config.use_huber_loss:
            self._loop_noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber(self.config.huber_k), base_loop_noise
            )
        else:
            self._loop_noise = base_loop_noise
        self._absolute_rotation_noise_default = (
            gtsam.noiseModel.Diagonal.Sigmas(self.config.absolute_rotation_noise_sigmas)
            if self.config.absolute_rotation_noise_sigmas is not None
            else None
        )

        self._frame_index = -1
        self._raw_pose = gtsam.Pose3.Identity()
        self._current_keyframe_pose: gtsam.Pose3 | None = None
        self._pending_keyframe_odometry = gtsam.Pose3.Identity()
        self._has_pending_keyframe_odometry = False

        self._raw_trajectory: list[gtsam.Pose3] = []
        self._gt_trajectory: list[gtsam.Pose3] = []
        self._raw_keyframe_trajectory: list[gtsam.Pose3] = []
        self._gt_keyframe_trajectory: list[gtsam.Pose3] = []
        # Accumulated loop-closure inliers for rerun visualization.
        self._loop_closure_inlier_points: list[tuple[int, np.ndarray, np.ndarray]] = []

    def shutdown(self) -> None:
        self.loop_manager.stop()

    def process_step(
        self,
        frame: StereoFrame,
        odometry: gtsam.Pose3 | None,
        absolute_rotation: gtsam.Rot3 | None = None,
        absolute_rotation_noise: np.ndarray | object | None = None,
        *,
        timestamp: float | None = None,  # reserved for future real-time gating
        ground_truth_pose: gtsam.Pose3 | None = None,
    ) -> SlamStepResult:
        """
        Process a single stereo frame with an associated odometry measurement.

        Args:
            frame: Rectified or raw stereo frame.
            odometry: Relative pose from the previous keyframe to the current frame.
            absolute_rotation: Optional absolute orientation measurement for the current frame.
            absolute_rotation_noise: Either a noise model or 3-vector of sigmas for the rotation prior.
            timestamp: Optional timestamp for future real-time handling.
            ground_truth_pose: Optional world pose, used only for evaluation/logging.
        """

        del timestamp  # placeholder until real-time scheduling is reintroduced

        self._frame_index += 1
        step_start = time.perf_counter()
        self.performance.start_step(self._frame_index)
        keyframe_depth_frame: tuple[int, StereoDepthFrame] | None = None

        with self.performance.time_section("odometry.integrate"):
            latest_pose = self._integrate_odometry(odometry)
        self._accumulate_keyframe_odometry(odometry)
        frontend_output: FrontendOutput | None = None
        is_keyframe = False
        rotation_measurement: tuple[gtsam.Rot3, object] | None = None
        if absolute_rotation is not None:
            noise_model = self._resolve_absolute_rotation_noise(absolute_rotation_noise)
            if noise_model is None:
                raise ValueError(
                    "Absolute rotation noise must be provided when supplying absolute rotations."
                )
            rotation_measurement = (absolute_rotation, noise_model)

        if ground_truth_pose is not None:
            self._gt_trajectory.append(ground_truth_pose)

        if self._current_keyframe_pose is None:
            with self.performance.time_section("frontend.total"):
                frontend_output = self.frontend.process(frame)
            self._record_frontend_timings(frontend_output.timings)
            pending_odometry = self._consume_pending_keyframe_odometry()
            if pending_odometry is not None:
                with self.performance.time_section("pose_graph.process_odometry"):
                    self.pose_graph.process_odometry(
                        pending_odometry,
                        self._odometry_noise,
                        frontend_output.feature_frame,
                        absolute_rotation=None if rotation_measurement is None else rotation_measurement[0],
                        absolute_rotation_noise=None if rotation_measurement is None else rotation_measurement[1],
                    )
            keyframe_depth_frame = (self.pose_graph.kf_idx, frontend_output.depth_frame)

            self._current_keyframe_pose = latest_pose
            self._raw_keyframe_trajectory.append(latest_pose)
            if ground_truth_pose is not None:
                self._gt_keyframe_trajectory.append(ground_truth_pose)

            is_keyframe = True
        else:
            relative_to_keyframe = self._current_keyframe_pose.inverse().compose(latest_pose)
            distance = np.linalg.norm(relative_to_keyframe.translation())
            rotation = np.linalg.norm(relative_to_keyframe.rotation().ypr())

            if (
                distance > self.config.keyframe_translation_threshold
                or rotation > self.config.keyframe_rotation_threshold
            ):
                with self.performance.time_section("frontend.total"):
                    frontend_output = self.frontend.process(frame)
                self._record_frontend_timings(frontend_output.timings)
                pending_odometry = self._consume_pending_keyframe_odometry()
                if pending_odometry is not None:
                    with self.performance.time_section("pose_graph.process_odometry"):
                        self.pose_graph.process_odometry(
                            pending_odometry,
                            self._odometry_noise,
                            frontend_output.feature_frame,
                            absolute_rotation=None if rotation_measurement is None else rotation_measurement[0],
                            absolute_rotation_noise=None if rotation_measurement is None else rotation_measurement[1],
                        )
                keyframe_depth_frame = (self.pose_graph.kf_idx, frontend_output.depth_frame)

                self._current_keyframe_pose = latest_pose
                self._raw_keyframe_trajectory.append(latest_pose)
                if ground_truth_pose is not None:
                    self._gt_keyframe_trajectory.append(ground_truth_pose)

                is_keyframe = True

        loop_closures_added = self._ingest_loop_closure_results()

        if is_keyframe:
            self._queue_loop_closure_candidates()
            # optionally attempt to fuse late-arriving closures after queuing new ones
            loop_closures_added += self._ingest_loop_closure_results()

        metrics: TrajectoryMetrics | None = None
        if is_keyframe or loop_closures_added > 0:
            with self.performance.time_section("optimization"):
                self.pose_graph.optimize()
            if self.logger is not None:
                with self.performance.time_section("logging.rerun"):
                    metrics = self.logger.log_step(
                        self._frame_index,
                        self.pose_graph,
                        self._gt_keyframe_trajectory,
                        self._raw_keyframe_trajectory,
                        keyframe_frame=keyframe_depth_frame,
                        loop_inlier_points=self._loop_closure_inlier_points,
                    )

        total_time = time.perf_counter() - step_start
        self.performance.record("total_step", total_time)
        snapshot = self.performance.end_step(total_time)

        return SlamStepResult(
            frame_index=self._frame_index,
            is_keyframe=is_keyframe,
            loop_closures_added=loop_closures_added,
            latest_pose=latest_pose,
            frontend_timings=frontend_output.timings if frontend_output else None,
            total_time=total_time,
            metrics=metrics,
            performance=snapshot,
        )

    def _integrate_odometry(self, odometry: gtsam.Pose3 | None) -> gtsam.Pose3:
        if odometry is not None:
            self._raw_pose = self._raw_pose.compose(odometry)

        self._raw_trajectory.append(self._raw_pose)
        return self._raw_pose

    def _accumulate_keyframe_odometry(self, odometry: gtsam.Pose3 | None) -> None:
        if odometry is None:
            return
        if self._has_pending_keyframe_odometry:
            self._pending_keyframe_odometry = self._pending_keyframe_odometry.compose(odometry)
        else:
            self._pending_keyframe_odometry = odometry
            self._has_pending_keyframe_odometry = True

    def _consume_pending_keyframe_odometry(self) -> gtsam.Pose3 | None:
        if not self._has_pending_keyframe_odometry:
            return None
        odometry = self._pending_keyframe_odometry
        self._pending_keyframe_odometry = gtsam.Pose3.Identity()
        self._has_pending_keyframe_odometry = False
        return odometry

    def _record_frontend_timings(self, timings: FrontendTimings) -> None:
        self.performance.record("frontend.rectify", timings.rectify)
        self.performance.record("frontend.depth", timings.depth)
        self.performance.record("frontend.features", timings.feature_detection)

    def _resolve_absolute_rotation_noise(
        self,
        noise: object | np.ndarray | None,
    ):
        if noise is None:
            return self._absolute_rotation_noise_default
        if hasattr(noise, "dim"):
            return noise
        sigmas = np.asarray(noise, dtype=float).reshape(3)
        return gtsam.noiseModel.Diagonal.Sigmas(sigmas)

    def _queue_loop_closure_candidates(self) -> None:
        total_start = time.perf_counter()
        detect_start = total_start
        loop_candidates = self.loop_detector.candidates(self.pose_graph, self.pose_graph.kf_idx)
        self.performance.record("loop_candidates.detect", time.perf_counter() - detect_start)

        if not loop_candidates:
            self.performance.record("loop_candidates.total", time.perf_counter() - total_start)
            return

        enqueue_start = time.perf_counter()
        candidates = list(reversed(loop_candidates))
        self.loop_manager.submit_candidates(candidates)
        self.performance.record("loop_candidates.enqueue", time.perf_counter() - enqueue_start)
        self.performance.record("loop_candidates.total", time.perf_counter() - total_start)

    def _ingest_loop_closure_results(self) -> int:
        with self.performance.time_section("loop_results.total"):
            poll_start = time.perf_counter()
            results = self.loop_manager.poll_results()
            self.performance.record("loop_results.poll", time.perf_counter() - poll_start)

            added = 0
            for result in results:
                with self.performance.time_section("loop_results.apply"):
                    points = self._add_loop_closure(result)
                if points is not None:
                    self._loop_closure_inlier_points.append(points)
                added += 1
            return added

    def _add_loop_closure(self, result: LoopClosureResult) -> tuple[int, np.ndarray, np.ndarray] | None:
        first_to_second = gtsam.Pose3(
            gtsam.Rot3(result.rotation_matrix), gtsam.Point3(result.translation)
        )

        inlier_points: tuple[int, np.ndarray, np.ndarray] | None = None
        frame = self.pose_graph.frames.get(result.first_idx)
        if frame is not None and result.inlier_matches.size > 0:
            match_indices = np.asarray(result.inlier_matches, dtype=np.int32)
            match_indices = np.atleast_2d(match_indices)

            if match_indices.ndim == 2 and match_indices.shape[1] >= 1:
                first_indices = match_indices[:, 0]
                keypoints_3d_src = frame.features.get("keypoints_3d")
                keypoints_color_src = frame.features.get("keypoints_color")

                if keypoints_3d_src is not None and keypoints_color_src is not None:
                    keypoints_3d = np.asarray(keypoints_3d_src)[first_indices]
                    keypoints_color = np.asarray(keypoints_color_src)[first_indices]
                    inlier_points = (result.first_idx, keypoints_3d, keypoints_color)

        with self.performance.time_section("pose_graph.add_loop_closure"):
            self.pose_graph.add_between_pose_factor(
                result.first_idx, result.second_idx, first_to_second, self._loop_noise
            )
        return inlier_points

def create_default_slam_system(
    *,
    calibration_matrix: np.ndarray,
    config: SlamConfig | None = None,
) -> StereoSlamSystem:
    """Factory that wires up the default stereo SLAM stack."""

    cfg = config or SlamConfig()

    pose_graph = GtsamPoseGraph(K=calibration_matrix)
    feature_matcher = create_matcher(cfg.feature_matcher)
    frontend = StereoFrontend(
        depth_estimator=SGBM(num_disparities=16 * 4, block_size=5, image_color="RGB"),
        feature_detector=feature_matcher,
        rectify_inputs=cfg.rectify_inputs,
    )
    loop_detector = ProximityLoopDetector(
        max_translation=cfg.proximity_max_translation,
        max_rotation=cfg.proximity_max_rotation,
        max_candidates=cfg.proximity_max_candidates,
        min_seperation=cfg.proximity_min_separation,
    )
    loop_manager = LoopClosureManager(
        min_inlier_count=cfg.loop_min_inliers,
        num_workers=cfg.loop_worker_count,
        matcher_type=cfg.feature_matcher,
    )
    logger: RerunLogger | None = None
    if cfg.enable_rerun_logging:
        logger = RerunLogger(
            app_id=cfg.rerun_app_id,
            tcp_address=cfg.rerun_tcp_address,
            enable_alignment=cfg.align_ground_truth,
        )

    return StereoSlamSystem(
        cfg,
        pose_graph=pose_graph,
        frontend=frontend,
        loop_detector=loop_detector,
        loop_manager=loop_manager,
        logger=logger,
    )
