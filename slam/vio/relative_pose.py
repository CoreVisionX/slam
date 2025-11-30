from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gtsam
import numpy as np

from slam.registration.registration import (
    FeatureFrame,
    MatchedFramePair,
    RectifiedStereoFrame,
)
from slam.registration.utils import solve_pnp

from .klt_tracker import TrackObservation


@dataclass
class RelativePnPInitializerConfig:
    min_matches_for_pnp: int = 6
    max_depth: float = 40.0


class RelativePnPInitializer:
    """Estimate relative poses between consecutive frames using PnP on tracked features."""

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
        self._prev_track_observations = None
        self._prev_frame_index = None
        self.results = []

    def reset(self, pose: gtsam.Pose3) -> None:
        self._world_pose = pose
        self._sequence_sample = None
        self._first_gt_pose = None
        self._prev_rectified_frame = None
        self._prev_track_observations = None
        self._prev_frame_index = None
        self.results = []

    def estimate_sequence_poses(
        self,
        rectified_frames: list[RectifiedStereoFrame],
        track_history: list[dict[int, TrackObservation]],
        sequence_sample: Any,
    ) -> list[dict[str, Any]]:
        self.reset_with_gt(sequence_sample)
        for frame_idx, rect_frame in enumerate(rectified_frames):
            track_obs = track_history[frame_idx]
            self.process_frame(
                frame_index=frame_idx,
                rectified_frame=rect_frame,
                track_observations=track_obs,
            )
        return list(self.results)

    def process_frame(
        self,
        *,
        frame_index: int,
        rectified_frame: RectifiedStereoFrame,
        track_observations: dict[int, TrackObservation],
        frame_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Incrementally process a frame and return the PnP result when available."""
        if self._world_pose is None:
            raise RuntimeError("RelativePnPInitializer.reset() must be called before processing frames.")

        if self._prev_track_observations is None:
            self._prev_track_observations = track_observations
            self._prev_rectified_frame = rectified_frame
            self._prev_frame_index = frame_index
            return None

        prev_obs = self._prev_track_observations
        prev_frame_idx = self._prev_frame_index
        prev_rectified = self._prev_rectified_frame

        if prev_frame_idx is None or prev_rectified is None:
            raise RuntimeError("Previous frame data missing; ensure frames are processed in order.")

        cfg = self.config
        sequence_sample = self._sequence_sample
        prev_world_pose = self._world_pose
        curr_obs = track_observations
        
        # Identify common tracks
        common_track_ids = sorted(set(prev_obs.keys()) & set(curr_obs.keys()))
        
        current_frame_id = frame_id
        if current_frame_id is None:
            if sequence_sample is not None:
                current_frame_id = sequence_sample.frame_ids[frame_index]
            else:
                current_frame_id = str(frame_index)

        if not common_track_ids:
            return self._record_failure(
                frame_index, current_frame_id, "no_tracks", 0,
                rectified_frame, track_observations
            )

        # Gather points and verify depth from previous observations
        prev_kps_list = []
        prev_depths_list = []
        curr_kps_list = []
        valid_track_ids_list = []

        for tid in common_track_ids:
            obs = prev_obs[tid]
            depth_val = obs.depth
            
            # Filter invalid depths
            if depth_val <= 0.0 or depth_val > cfg.max_depth or not np.isfinite(depth_val):
                continue
                
            prev_kps_list.append(obs.keypoint)
            prev_depths_list.append(depth_val)
            curr_kps_list.append(curr_obs[tid].keypoint)
            valid_track_ids_list.append(tid)

        if len(prev_kps_list) < cfg.min_matches_for_pnp:
            return self._record_failure(
                frame_index, current_frame_id, "insufficient_tracks", 
                len(prev_kps_list), rectified_frame, track_observations
            )

        # Convert to numpy arrays
        prev_kps = np.array(prev_kps_list, dtype=np.float32)
        prev_depths = np.array(prev_depths_list, dtype=np.float32)
        curr_kps = np.array(curr_kps_list, dtype=np.float32)
        valid_track_ids = np.array(valid_track_ids_list, dtype=int)

        # Unproject previous points to 3D using previous frame intrinsics (Rectified)
        K = prev_rectified.calibration.K_left_rect
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        z = prev_depths
        x = (prev_kps[:, 0] - cx) * z / fx
        y = (prev_kps[:, 1] - cy) * z / fy
        prev_points_3d = np.stack([x, y, z], axis=1).astype(np.float32)

        # Create dummy colors (required by MatchedFramePair property accessors usually)
        # Sampling from the image is safer than creating zeros
        h, w = prev_rectified.left_rect.shape[:2]
        u_clip = np.clip(np.round(prev_kps[:, 0]), 0, w - 1).astype(int)
        v_clip = np.clip(np.round(prev_kps[:, 1]), 0, h - 1).astype(int)
        prev_colors = prev_rectified.left_rect[v_clip, u_clip]

        # Construct FeatureFrames
        features1 = {
            "keypoints": prev_kps,
            "keypoints_depth": prev_depths,
            "keypoints_3d": prev_points_3d,
            "keypoints_color": prev_colors,
            "image_size": (h, w)
        }
        
        first_frame = FeatureFrame(
            left=None, right=None,
            left_rect=None, right_rect=None,
            left_depth=None, left_depth_xyz=None,
            calibration=prev_rectified.calibration,
            features=features1
        )

        features2 = {
            "keypoints": curr_kps,
            "image_size": rectified_frame.left_rect.shape[:2]
        }
        
        second_frame = FeatureFrame(
            left=None, right=None,
            left_rect=None, right_rect=None,
            left_depth=None, left_depth_xyz=None,
            calibration=rectified_frame.calibration,
            features=features2
        )

        match_indices = np.arange(len(prev_kps), dtype=int)
        matches = np.stack([match_indices, match_indices], axis=1)
        matched_pair = MatchedFramePair(
            first=first_frame,
            second=second_frame,
            matches=matches,
        )

        try:
            # solve_pnp typically expects 3D points in 'first' and 2D in 'second'
            relative_pose_s0, inlier_pair = solve_pnp(matched_pair)
            
            # The result of solve_pnp is usually Frame2 w.r.t Frame1 (or Camera w.r.t World).
            # We need to account for IMU-Camera extrinsics if provided.
            # Assuming `imu_from_left` exists on calibration (common in VIO/SLAM setups)
            # If not present in the dataclass provided, this line might need adjustment,
            # but based on the previous context it seems expected.
            if hasattr(prev_rectified.calibration, 'imu_from_left'):
                imu_from_left = prev_rectified.calibration.imu_from_left
                relative_pose = imu_from_left * relative_pose_s0 * imu_from_left.inverse()
            else:
                # Fallback if extrinsics are not available
                relative_pose = relative_pose_s0

        except Exception as exc:  # noqa: BLE001
            return self._record_failure(
                frame_index, current_frame_id, "pnp_failed", 
                len(prev_kps), rectified_frame, track_observations, str(exc)
            )

        estimated_pose = prev_world_pose.compose(relative_pose)
        self._world_pose = estimated_pose
        
        ground_truth_pose = None
        pose_error = None
        if sequence_sample is not None and self._first_gt_pose is not None:
            # Assuming world_poses is a list of gtsam.Pose3
            gt_pose_now = sequence_sample.world_poses[frame_index]
            # Adjust GT to be relative to the first frame (start at 0,0,0)
            ground_truth_pose = (self._first_gt_pose.inverse() * gt_pose_now)
            pose_error = ground_truth_pose.between(estimated_pose)

        result = {
            "frame_index": frame_index,
            "frame_id": current_frame_id,
            "status": "success",
            "active_track_count": len(prev_kps),
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
        self._update_previous_frame(rectified_frame, track_observations, frame_index)
        return result

    def _record_failure(
        self, 
        frame_index: int, 
        frame_id: str, 
        status: str, 
        count: int,
        rectified_frame: RectifiedStereoFrame,
        track_observations: dict[int, TrackObservation],
        error_msg: str | None = None
    ) -> dict[str, Any]:
        result = {
            "frame_index": frame_index,
            "frame_id": frame_id,
            "status": status,
            "active_track_count": count,
        }
        if error_msg:
            result["error"] = error_msg
        
        if self.store_results:
            self.results.append(result)
        
        self._update_previous_frame(rectified_frame, track_observations, frame_index)
        return result

    def _update_previous_frame(
        self,
        rectified_frame: RectifiedStereoFrame,
        track_observations: dict[int, TrackObservation],
        frame_index: int,
    ) -> None:
        self._prev_rectified_frame = rectified_frame
        self._prev_track_observations = track_observations
        self._prev_frame_index = frame_index


__all__ = [
    "RelativePnPInitializer",
    "RelativePnPInitializerConfig",
]