from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from slam.registration.registration import RectifiedStereoFrame, StereoDepthFrame

from .feature_utils import extract_keypoint_attributes


@dataclass
class TrackObservation:
    keypoint: np.ndarray
    depth: float


@dataclass
class FeatureTrack:
    track_id: int
    anchor_frame: int
    anchor_keypoint: np.ndarray
    anchor_depth: float
    anchor_point3: np.ndarray
    anchor_color: np.ndarray
    observations: dict[int, TrackObservation] = field(default_factory=dict)
    observation_frames: list[int] = field(default_factory=list)
    active: bool = True

    def add_observation(self, frame_idx: int, point: np.ndarray, depth: float) -> None:
        self.observations[frame_idx] = TrackObservation(
            keypoint=np.asarray(point, dtype=np.float32),
            depth=float(depth),
        )
        if not self.observation_frames or self.observation_frames[-1] != frame_idx:
            self.observation_frames.append(frame_idx)


@dataclass
class KLTTrackerConfig:
    max_feature_count: int = 1024
    refill_feature_ratio: float = 0.8
    feature_suppression_radius: float = 8.0
    fast_threshold: int = 25
    fast_nonmax: bool = True
    fast_border: int = 12
    lk_win_size: tuple[int, int] = (15, 15)
    lk_max_level: int = 5
    lk_max_iterations: int = 40
    lk_epsilon: float = 0.01
    lk_min_eig_threshold: float = 1e-3
    max_depth: float = 40.0


class KLTFeatureTracker:
    """KLT feature tracker with optional feature refill."""

    def __init__(
        self,
        *,
        max_feature_count: int = 1024,
        refill_feature_ratio: float = 0.8,
        feature_suppression_radius: float = 8.0,
        fast_threshold: int = 25,
        fast_nonmax: bool = True,
        fast_border: int = 12,
        lk_win_size: tuple[int, int] = (15, 15),
        lk_max_level: int = 5,
        lk_max_iterations: int = 40,
        lk_epsilon: float = 0.01,
        lk_min_eig_threshold: float = 1e-3,
        max_depth: float = 40.0,
    ) -> None:
        self.config = KLTTrackerConfig(
            max_feature_count=max_feature_count,
            refill_feature_ratio=refill_feature_ratio,
            feature_suppression_radius=feature_suppression_radius,
            fast_threshold=fast_threshold,
            fast_nonmax=fast_nonmax,
            fast_border=fast_border,
            lk_win_size=tuple(lk_win_size),
            lk_max_level=lk_max_level,
            lk_max_iterations=lk_max_iterations,
            lk_epsilon=lk_epsilon,
            lk_min_eig_threshold=lk_min_eig_threshold,
            max_depth=max_depth,
        )
        self._lk_params = dict(
            winSize=tuple(int(v) for v in self.config.lk_win_size),
            maxLevel=self.config.lk_max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.config.lk_max_iterations,
                self.config.lk_epsilon,
            ),
            minEigThreshold=self.config.lk_min_eig_threshold,
        )
        self._refill_threshold = int(np.floor(self.config.refill_feature_ratio * self.config.max_feature_count))
        self.reset()

    def reset(self) -> None:
        """Reset tracker state for a new sequence."""
        self.tracks: dict[int, FeatureTrack] = {}
        self.track_history: list[dict[int, TrackObservation]] = []
        self._next_track_id = 0
        self._prev_gray: np.ndarray | None = None
        self._prev_points = np.empty((0, 1, 2), dtype=np.float32)
        self._prev_ids = np.empty(0, dtype=int)

    def _filter_keypoints_by_distance(
        self,
        candidate_points: np.ndarray,
        existing_points: np.ndarray,
        min_radius: float,
    ) -> np.ndarray:
        if candidate_points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if existing_points.size == 0:
            return candidate_points.astype(np.float32)

        kept: list[np.ndarray] = []
        current = existing_points.reshape(-1, 2).astype(np.float32)
        min_radius_sq = float(min_radius * min_radius)
        for point in candidate_points.astype(np.float32):
            if current.size:
                deltas = current - point
                if np.min(np.sum(deltas * deltas, axis=1)) < min_radius_sq:
                    continue
            kept.append(point)
            if current.size:
                current = np.vstack([current, point.reshape(1, 2)])
            else:
                current = point.reshape(1, 2)
        if not kept:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(kept, dtype=np.float32)

    def _sample_depth_at_point(self, depth_frame: StereoDepthFrame, point: np.ndarray) -> float:
        depth_map = depth_frame.left_depth
        if depth_map.size == 0:
            return float("nan")
        h, w = depth_map.shape
        u = float(point[0])
        v = float(point[1])
        if not (0 <= u < w and 0 <= v < h):
            return float("nan")
        col = int(np.clip(np.round(u), 0, w - 1))
        row = int(np.clip(np.round(v), 0, h - 1))
        depth_value = float(depth_map[row, col])
        if not np.isfinite(depth_value) or depth_value <= 0.0:
            return float("nan")
        return depth_value

    def _detect_fast_keypoints(
        self,
        gray_image: np.ndarray,
        max_features: int,
        threshold: int,
        nonmax: bool,
        border: int,
    ) -> np.ndarray:
        detector = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax)
        keypoints = detector.detect(gray_image)
        if not keypoints:
            raise RuntimeError("FAST detector did not return any keypoints.")

        points = cv2.KeyPoint_convert(keypoints)
        responses = np.array([kp.response for kp in keypoints], dtype=np.float32)

        if border > 0:
            h, w = gray_image.shape
            inside = (
                (points[:, 0] >= border)
                & (points[:, 0] < w - border)
                & (points[:, 1] >= border)
                & (points[:, 1] < h - border)
            )
            points = points[inside]
            responses = responses[inside]

        if points.size == 0:
            raise RuntimeError("All FAST keypoints were rejected by the border filter.")

        if points.shape[0] > max_features:
            order = np.argsort(responses)[::-1][:max_features]
            points = points[order]

        return points.astype(np.float32)

    def track_frame(
        self,
        rectified_frame: RectifiedStereoFrame,
        depth_frame: StereoDepthFrame,
    ) -> dict[int, TrackObservation]:
        """Track features for a single frame and update state."""
        frame_idx = len(self.track_history)
        cfg = self.config
        gray = cv2.cvtColor(rectified_frame.left_rect, cv2.COLOR_RGB2GRAY)
        frame_obs: dict[int, TrackObservation] = {}
        tracked_points: list[np.ndarray] = []
        tracked_ids: list[int] = []

        if self._prev_gray is not None and self._prev_points.size and self._prev_ids.size:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray,
                gray,
                self._prev_points,
                None,
                **self._lk_params,
            )
            if next_points is None or status is None:
                status_mask = np.zeros(self._prev_ids.shape[0], dtype=bool)
                next_points_arr = np.zeros((self._prev_ids.shape[0], 2), dtype=np.float32)
            else:
                status_mask = status.reshape(-1).astype(bool)
                next_points_arr = next_points.reshape(-1, 2).astype(np.float32)

            for idx, track_id in enumerate(self._prev_ids):
                if idx >= status_mask.size or not status_mask[idx]:
                    self.tracks[track_id].active = False
                    continue

                point = next_points_arr[idx]
                depth_value = self._sample_depth_at_point(depth_frame, point)
                track = self.tracks[track_id]
                track.add_observation(frame_idx, point, depth_value)
                frame_obs[track_id] = track.observations[frame_idx]
                tracked_points.append(point)
                tracked_ids.append(track_id)

        if tracked_points:
            existing_points = np.asarray(tracked_points, dtype=np.float32)
        else:
            existing_points = np.empty((0, 2), dtype=np.float32)

        def add_new_tracks(budget: int) -> tuple[list[int], np.ndarray]:
            nonlocal existing_points
            if budget <= 0:
                return [], np.empty((0, 2), dtype=np.float32)

            detection_quota = max(cfg.max_feature_count, budget * 2)
            try:
                candidate_points = self._detect_fast_keypoints(
                    gray,
                    detection_quota,
                    cfg.fast_threshold,
                    cfg.fast_nonmax,
                    cfg.fast_border,
                )
            except RuntimeError:
                if frame_idx == 0:
                    raise
                return [], np.empty((0, 2), dtype=np.float32)

            filtered_candidates = self._filter_keypoints_by_distance(
                candidate_points,
                existing_points,
                cfg.feature_suppression_radius,
            )
            if filtered_candidates.size == 0:
                return [], np.empty((0, 2), dtype=np.float32)

            attributes = extract_keypoint_attributes(depth_frame, filtered_candidates, max_depth=cfg.max_depth)
            new_keypoints = attributes["keypoints"]
            if new_keypoints.size == 0:
                return [], np.empty((0, 2), dtype=np.float32)

            depths = attributes["keypoints_depth"]
            xyz = attributes["keypoints_3d"]
            colors = attributes["keypoints_color"]

            take = min(budget, new_keypoints.shape[0])
            new_ids: list[int] = []
            new_points_list: list[np.ndarray] = []
            for idx_new in range(take):
                keypoint = new_keypoints[idx_new]
                depth_value = float(depths[idx_new])
                point3 = np.asarray(xyz[idx_new], dtype=np.float64)
                color = np.asarray(colors[idx_new])

                track = FeatureTrack(
                    track_id=self._next_track_id,
                    anchor_frame=frame_idx,
                    anchor_keypoint=keypoint.astype(np.float32),
                    anchor_depth=depth_value,
                    anchor_point3=point3,
                    anchor_color=color,
                )
                track.add_observation(frame_idx, keypoint, depth_value)
                self.tracks[self._next_track_id] = track
                new_ids.append(self._next_track_id)
                new_points_list.append(keypoint.astype(np.float32))
                frame_obs[self._next_track_id] = track.observations[frame_idx]
                self._next_track_id += 1

            if new_points_list:
                new_points_arr = np.asarray(new_points_list, dtype=np.float32)
                existing_points = (
                    np.vstack([existing_points, new_points_arr]) if existing_points.size else new_points_arr
                )
            else:
                new_points_arr = np.empty((0, 2), dtype=np.float32)
            return new_ids, new_points_arr

        need_refill = len(tracked_ids) <= self._refill_threshold
        budget = cfg.max_feature_count - len(tracked_ids) if need_refill else 0
        new_ids, new_points = add_new_tracks(budget)
        if new_ids:
            tracked_ids.extend(new_ids)
            if new_points.size:
                tracked_points.extend(new_points.reshape(-1, 2))

        active_ids = np.asarray(tracked_ids, dtype=int) if tracked_ids else np.empty(0, dtype=int)
        if tracked_points:
            stacked_points = np.asarray(tracked_points, dtype=np.float32).reshape(-1, 1, 2)
        else:
            stacked_points = np.empty((0, 1, 2), dtype=np.float32)

        for track_id in active_ids:
            self.tracks[track_id].active = True

        self.track_history.append(frame_obs)
        self._prev_gray = gray
        self._prev_points = stacked_points
        self._prev_ids = active_ids
        return frame_obs

    def track(
        self,
        rectified_frames: list[RectifiedStereoFrame],
        depth_frames: list[StereoDepthFrame],
    ) -> tuple[list[dict[int, TrackObservation]], dict[int, FeatureTrack]]:
        if len(rectified_frames) != len(depth_frames):
            raise ValueError("Number of rectified frames and depth frames must match.")

        self.reset()
        for rectified_frame, depth_frame in zip(rectified_frames, depth_frames):
            self.track_frame(rectified_frame, depth_frame)
        return list(self.track_history), dict(self.tracks)


__all__ = [
    "FeatureTrack",
    "KLTFeatureTracker",
    "KLTTrackerConfig",
    "TrackObservation",
]
