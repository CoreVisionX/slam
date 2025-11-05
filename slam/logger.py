"""Rerun logging utilities for the SLAM system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gtsam
import numpy as np
import rerun as rr
from gtsam.symbol_shorthand import X

from backend.pose_graph import GtsamPoseGraph
from slam.alignment import compute_umeyama_alignment_pose, pose_translation_to_array
from viz import rr_log_graph_edges, rr_log_trajectory


@dataclass(slots=True)
class TrajectoryMetrics:
    translation_ate: float
    rotation_ate_deg: float


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
            rr.connect_tcp(tcp_address)

        self._enable_alignment = enable_alignment

    def log_step(
        self,
        frame_index: int,
        pose_graph: GtsamPoseGraph,
        gt_keyframe_trajectory: Sequence[gtsam.Pose3],
        raw_keyframe_trajectory: Sequence[gtsam.Pose3],
    ) -> TrajectoryMetrics | None:
        """Log current SLAM status. Returns metrics when available."""

        if len(gt_keyframe_trajectory) < 2:
            return None

        estimated: list[gtsam.Pose3] = []
        for idx in range(1, pose_graph.kf_idx + 1):
            key = X(idx)
            if pose_graph.values.exists(key):
                estimated.append(pose_graph.values.atPose3(key))
            else:
                estimated.clear()
                break

        if len(estimated) != len(gt_keyframe_trajectory):
            return None

        rr.set_time_sequence("frame", sequence=frame_index)

        gt_for_logging = list(gt_keyframe_trajectory)
        if self._enable_alignment:
            alignment_pose = compute_umeyama_alignment_pose(gt_for_logging, estimated)
            if alignment_pose is not None:
                gt_for_logging = [alignment_pose.compose(pose) for pose in gt_for_logging]

        rr_log_trajectory("gt_keyframe_trajectory", gt_for_logging, color=(0, 255, 0))
        rr_log_trajectory("raw_keyframe_trajectory", raw_keyframe_trajectory, color=(0, 0, 255))
        rr_log_graph_edges(path="graph", nodes=pose_graph.values, graph=pose_graph.graph)

        metrics = self._compute_metrics(gt_for_logging, estimated)
        rr.log("/translation/ate", rr.Scalar(metrics.translation_ate))
        rr.log("/rotation/ate", rr.Scalar(metrics.rotation_ate_deg))

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

        rotation_errors = [
            np.linalg.norm(gt_pose.rotation().ypr() - est_pose.rotation().ypr())
            for gt_pose, est_pose in zip(aligned_gt, estimated)
        ]
        rotation_ate_deg = float(np.rad2deg(np.mean(rotation_errors)))

        return TrajectoryMetrics(
            translation_ate=translation_ate,
            rotation_ate_deg=rotation_ate_deg,
        )

