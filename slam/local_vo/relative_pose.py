from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gtsam
import numpy as np

from registration.registration import MatchedFramePair, RectifiedStereoFrame, StereoDepthFrame
from registration.utils import solve_pnp

from .feature_utils import build_feature_frame, extract_keypoint_attributes, make_feature_frame_for_view
from .klt_tracker import TrackObservation


@dataclass
class RelativePnPInitializerConfig:
    min_matches_for_pnp: int = 6
    max_depth: float = 40.0


class RelativePnPInitializer:
    """Estimate relative poses between consecutive frames using PnP."""

    def __init__(
        self,
        *,
        min_matches_for_pnp: int = 6,
        max_depth: float = 40.0,
        store_results: bool = True,
    ) -> None:
        self.config = RelativePnPInitializerConfig(
            min_matches_for_pnp=min_matches_for_pnp,
            max_depth=max_depth,
        )
        self._sequence_sample: Any | None = None
        self._first_gt_pose: gtsam.Pose3 | None = None
        self._world_pose: gtsam.Pose3 | None = None
        self._prev_rectified_frame: RectifiedStereoFrame | None = None
        self._prev_depth_frame: StereoDepthFrame | None = None
        self._prev_track_observations: dict[int, TrackObservation] | None = None
        self._prev_frame_index: int | None = None
        self.results: list[dict[str, Any]] = []
        self.store_results = store_results

    def reset_with_gt(self, sequence_sample: Any) -> None:
        """Reset state so poses can be estimated incrementally."""
        self._sequence_sample = sequence_sample
        first_pose = gtsam.Pose3(sequence_sample.world_poses[0].rotation(), np.zeros(3))
        self._first_gt_pose = first_pose
        self._world_pose = gtsam.Pose3(sequence_sample.world_poses[0].rotation(), np.zeros(3))
        self._prev_rectified_frame = None
        self._prev_depth_frame = None
        self._prev_track_observations = None
        self._prev_frame_index = None
        self.results = []

    def reset(self, pose: gtsam.Pose3) -> None:
        self._world_pose = pose
        self._sequence_sample = None
        self._first_gt_pose = None
        self._prev_rectified_frame = None
        self._prev_depth_frame = None
        self._prev_track_observations = None
        self._prev_frame_index = None
        self.results = []

    def estimate_sequence_poses(
        self,
        rectified_frames: list[RectifiedStereoFrame],
        depth_frames: list[StereoDepthFrame],
        track_history: list[dict[int, TrackObservation]],
        sequence_sample: Any,
    ) -> list[dict[str, Any]]:
        self.reset_with_gt(sequence_sample)
        for frame_idx, (rect_frame, depth_frame) in enumerate(zip(rectified_frames, depth_frames)):
            track_obs = track_history[frame_idx]
            self.process_frame(
                frame_index=frame_idx,
                rectified_frame=rect_frame,
                depth_frame=depth_frame,
                track_observations=track_obs,
            )
        return list(self.results)

    def process_frame(
        self,
        *,
        frame_index: int,
        rectified_frame: RectifiedStereoFrame,
        depth_frame: StereoDepthFrame,
        track_observations: dict[int, TrackObservation],
        frame_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Incrementally process a frame and return the PnP result when available."""
        if self._world_pose is None:
            raise RuntimeError("RelativePnPInitializer.reset() must be called before processing frames.")

        if self._prev_track_observations is None:
            self._prev_track_observations = track_observations
            self._prev_rectified_frame = rectified_frame
            self._prev_depth_frame = depth_frame
            self._prev_frame_index = frame_index
            return None

        prev_obs = self._prev_track_observations
        prev_frame_idx = self._prev_frame_index
        prev_depth = self._prev_depth_frame
        prev_rectified = self._prev_rectified_frame

        if prev_frame_idx is None or prev_depth is None or prev_rectified is None:
            raise RuntimeError("Previous frame data missing; ensure frames are processed in order.")

        cfg = self.config
        sequence_sample = self._sequence_sample
        prev_world_pose = self._world_pose
        curr_obs = track_observations
        common_track_ids = sorted(set(prev_obs.keys()) & set(curr_obs.keys()))
        
        current_frame_id = frame_id
        if current_frame_id is None:
            if sequence_sample is not None:
                current_frame_id = sequence_sample.frame_ids[frame_index]
            else:
                current_frame_id = str(frame_index)

        if not common_track_ids:
            result = {
                "frame_index": frame_index,
                "frame_id": current_frame_id,
                "status": "no_tracks",
                "active_track_count": 0,
            }
            if self.store_results:
                self.results.append(result)
            self._update_previous_frame(rectified_frame, depth_frame, track_observations, frame_index)
            return result

        prev_points = np.asarray(
            [prev_obs[track_id].keypoint for track_id in common_track_ids],
            dtype=np.float32,
        )
        curr_points = np.asarray(
            [curr_obs[track_id].keypoint for track_id in common_track_ids],
            dtype=np.float32,
        )

        attributes = extract_keypoint_attributes(
            prev_depth,
            prev_points,
            max_depth=cfg.max_depth,
        )
        valid_mask = attributes["valid_mask"]
        if not np.any(valid_mask):
            result = {
                "frame_index": frame_index,
                "frame_id": current_frame_id,
                "status": "no_valid_depth",
                "active_track_count": len(common_track_ids),
            }
            if self.store_results:
                self.results.append(result)
            self._update_previous_frame(rectified_frame, depth_frame, track_observations, frame_index)
            return result

        prev_valid = attributes["keypoints"]
        curr_valid = curr_points[valid_mask]
        valid_track_ids = np.asarray(common_track_ids, dtype=int)[valid_mask]
        if prev_valid.shape[0] < cfg.min_matches_for_pnp:
            result = {
                "frame_index": frame_index,
                "frame_id": current_frame_id,
                "status": "insufficient_tracks",
                "active_track_count": prev_valid.shape[0],
            }
            if self.store_results:
                self.results.append(result)
            self._update_previous_frame(rectified_frame, depth_frame, track_observations, frame_index)
            return result

        first_frame = build_feature_frame(prev_depth, attributes)
        second_frame = make_feature_frame_for_view(rectified_frame, curr_valid)
        match_indices = np.arange(prev_valid.shape[0], dtype=int)
        matches = np.stack([match_indices, match_indices], axis=1)
        matched_pair = MatchedFramePair(
            first=first_frame,
            second=second_frame,
            matches=matches,
        )

        try:
            relative_pose_s0, inlier_pair = solve_pnp(matched_pair)
            T_B_from_S0 = prev_rectified.calibration.T_B_from_S0
            relative_pose = T_B_from_S0 * relative_pose_s0 * T_B_from_S0.inverse()
        except Exception as exc:  # noqa: BLE001
            result = {
                "frame_index": frame_index,
                "frame_id": current_frame_id,
                "status": "pnp_failed",
                "active_track_count": prev_valid.shape[0],
                "error": str(exc),
            }
            if self.store_results:
                self.results.append(result)
            self._update_previous_frame(rectified_frame, depth_frame, track_observations, frame_index)
            return result

        estimated_pose = prev_world_pose.compose(relative_pose)
        self._world_pose = estimated_pose
        
        ground_truth_pose = None
        pose_error = None
        if sequence_sample is not None and self._first_gt_pose is not None:
            ground_truth_pose = (self._first_gt_pose.inverse() * sequence_sample.world_poses[frame_index])
            pose_error = ground_truth_pose.between(estimated_pose)

        result = {
            "frame_index": frame_index,
            "frame_id": current_frame_id,
            "status": "success",
            "active_track_count": prev_valid.shape[0],
            "matches_before_filter": len(common_track_ids),
            "matches_after_filter": inlier_pair.matches.shape[0],
            "estimated_pose": estimated_pose,
            "ground_truth_pose": ground_truth_pose,
            "pose_error": pose_error,
            "matched_pair": inlier_pair,
            "track_ids": valid_track_ids,
            "relative_pose": relative_pose,
            "anchor_frame_index": prev_frame_idx,
        }
        if self.store_results:
            self.results.append(result)
        self._update_previous_frame(rectified_frame, depth_frame, track_observations, frame_index)
        return result

    def _update_previous_frame(
        self,
        rectified_frame: RectifiedStereoFrame,
        depth_frame: StereoDepthFrame,
        track_observations: dict[int, TrackObservation],
        frame_index: int,
    ) -> None:
        self._prev_rectified_frame = rectified_frame
        self._prev_depth_frame = depth_frame
        self._prev_track_observations = track_observations
        self._prev_frame_index = frame_index


__all__ = [
    "RelativePnPInitializer",
    "RelativePnPInitializerConfig",
]
