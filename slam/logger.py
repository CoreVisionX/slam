"""Rerun logging utilities for the SLAM system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gtsam
import numpy as np
import rerun as rr
from gtsam.symbol_shorthand import X

from backend.pose_graph import GtsamPoseGraph
from registration.registration import StereoDepthFrame
from slam.alignment import (
    apply_similarity_transform,
    compute_umeyama_alignment_pose,
    pose_translation_to_array,
)
from viz import rr_log_graph_edges, rr_log_map_points, rr_log_pose, rr_log_trajectory


@dataclass(slots=True)
class TrajectoryMetrics:
    translation_ate: float
    rotation_ate_deg: float
    total_distance: float
    translation_ate_pct: float


class RerunLogger:
    """Log SLAM state to Rerun."""

    def __init__(
        self,
        *,
        app_id: str = "slam",
        tcp_address: str | None = None,
        enable_alignment: bool = True,
    ) -> None:
        rr.init(app_id)
        if tcp_address:
            rr.connect_grpc(tcp_address)

        self._enable_alignment = enable_alignment
        self._logged_keyframes: set[int] = set()
        self._map_points_world: list[np.ndarray] = []
        self._map_points_color: list[np.ndarray] = []
        self._max_points_per_keyframe = 50_000
        self._rng = np.random.default_rng()

    def log_step(
        self,
        frame_index: int,
        pose_graph: GtsamPoseGraph,
        gt_keyframe_trajectory: Sequence[gtsam.Pose3],
        raw_keyframe_trajectory: Sequence[gtsam.Pose3],
        *,
        keyframe_frame: tuple[int, StereoDepthFrame] | None = None,
        loop_inlier_points: Sequence[tuple[int, np.ndarray, np.ndarray]] | None = None,
    ) -> TrajectoryMetrics | None:
        """Log current SLAM status. Returns metrics when available."""

        rr.set_time("frame", sequence=frame_index)

        rr_log_trajectory("raw_keyframe_trajectory", raw_keyframe_trajectory, color=(0, 0, 255))
        rr_log_graph_edges(path="graph", nodes=pose_graph.values, graph=pose_graph.graph)
        new_points_logged = False
        if keyframe_frame is not None:
            kf_idx, frame = keyframe_frame
            try:
                pose = pose_graph.get_pose(kf_idx)
            except RuntimeError:
                pass
            else:
                rr_log_pose("keyframes/current", pose, frame)
                new_points_logged = self._accumulate_map_points(kf_idx, pose, frame)
        if loop_inlier_points:
            rr_log_map_points(
                "loop_closure/inliers",
                pose_graph,
                list(loop_inlier_points),
                height_colormap=True,
            )

        if new_points_logged:
            self._log_global_point_cloud()

        metrics: TrajectoryMetrics | None = None
        if len(gt_keyframe_trajectory) >= 2:
            estimated: list[gtsam.Pose3] = []
            for idx in range(1, pose_graph.kf_idx + 1):
                key = X(idx)
                if pose_graph.values.exists(key):
                    estimated.append(pose_graph.values.atPose3(key))
                else:
                    estimated.clear()
                    break

            if len(estimated) == len(gt_keyframe_trajectory) and estimated:
                gt_for_logging = list(gt_keyframe_trajectory)
                if self._enable_alignment:
                    alignment = compute_umeyama_alignment_pose(gt_for_logging, estimated)
                    if alignment is not None:
                        rotation, translation, scale = alignment
                        gt_for_logging = [
                            apply_similarity_transform(pose, rotation, translation, scale)
                            for pose in gt_for_logging
                        ]

                rr_log_trajectory("gt_keyframe_trajectory", gt_for_logging, color=(0, 255, 0))
                metrics = self._compute_metrics(gt_for_logging, estimated)
                rr.log("/translation/ate", rr.Scalars(metrics.translation_ate))
                rr.log("/translation/total_distance", rr.Scalars(metrics.total_distance))
                rr.log(
                    "/translation/ate_percentage",
                    rr.Scalars(metrics.translation_ate_pct),
                )
                rr.log("/rotation/ate", rr.Scalars(metrics.rotation_ate_deg))

        return metrics

    @staticmethod
    def _compute_metrics(
        aligned_gt: Sequence[gtsam.Pose3], estimated: Sequence[gtsam.Pose3]
    ) -> TrajectoryMetrics:
        translation_errors = [
            np.linalg.norm(
                pose_translation_to_array(gt_pose) - pose_translation_to_array(est_pose)
            )
            for gt_pose, est_pose in zip(aligned_gt, estimated)
        ]
        translation_ate = float(np.mean(translation_errors))

        gt_positions = np.array([pose_translation_to_array(pose) for pose in aligned_gt])
        if len(gt_positions) >= 2:
            segment_lengths = np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)
            total_distance = float(np.sum(segment_lengths))
        else:
            total_distance = 0.0
        translation_ate_pct = (
            float((translation_ate / total_distance) * 100.0) if total_distance > 1e-9 else 0.0
        )

        rotation_errors = [
            np.linalg.norm(gt_pose.rotation().ypr() - est_pose.rotation().ypr())
            for gt_pose, est_pose in zip(aligned_gt, estimated)
        ]
        rotation_ate_deg = float(np.rad2deg(np.mean(rotation_errors)))

        return TrajectoryMetrics(
            translation_ate=translation_ate,
            rotation_ate_deg=rotation_ate_deg,
            total_distance=total_distance,
            translation_ate_pct=translation_ate_pct,
        )

    def _accumulate_map_points(
        self,
        kf_idx: int,
        pose: gtsam.Pose3,
        frame: StereoDepthFrame,
    ) -> bool:
        if kf_idx in self._logged_keyframes:
            return False

        left_depth = getattr(frame, "left_depth", None)
        left_depth_xyz = getattr(frame, "left_depth_xyz", None)
        left_rect = getattr(frame, "left_rect", None)

        if left_depth is None or left_depth_xyz is None or left_rect is None:
            return False

        valid_mask = np.isfinite(left_depth) & (left_depth > 0.0)
        valid_mask &= np.all(np.isfinite(left_depth_xyz), axis=2)
        if not np.any(valid_mask):
            return False

        points = left_depth_xyz[valid_mask]
        colors = left_rect[valid_mask]
        if points.size == 0:
            return False

        if colors.ndim == 1:
            colors = np.repeat(colors[:, None], 3, axis=1)
        elif colors.shape[1] == 1:
            colors = np.repeat(colors, 3, axis=1)

        if points.shape[0] > self._max_points_per_keyframe:
            idx = self._rng.choice(points.shape[0], self._max_points_per_keyframe, replace=False)
            points = points[idx]
            colors = colors[idx]

        points = points.astype(np.float32, copy=False)
        colors = colors.astype(np.float32, copy=False)
        world_points = pose.transformFrom(points.T).T.astype(np.float32, copy=False)

        self._map_points_world.append(world_points)
        self._map_points_color.append(colors)
        self._logged_keyframes.add(kf_idx)
        return True

    def _log_global_point_cloud(self) -> None:
        if not self._map_points_world:
            return

        world_points = np.concatenate(self._map_points_world, axis=0)
        colors = np.concatenate(self._map_points_color, axis=0)
        rr.log(
            "map/global_points",
            rr.Points3D(world_points, colors=np.clip(colors, 0.0, 255.0), radii=[0.01]),
        )
