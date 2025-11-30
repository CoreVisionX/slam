from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from line_profiler import profile

from slam.registration.registration import RectifiedStereoFrame
from slam.vio import stereo_matching

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
    

    
    # Optical Flow settings
    lk_win_size: tuple[int, int] = (15, 15)
    lk_max_level: int = 5
    lk_max_iterations: int = 40
    lk_epsilon: float = 0.01
    lk_min_eig_threshold: float = 1e-3
    
    # RANSAC / Verification settings
    stereo_ransac_threshold: float = 2.0
    stereo_max_y_diff: float = 2.0
    min_disparity: float = 0.1
    max_depth: float = 40.0


    # Good Features To Track (GFTT) settings
    gftt_quality_level: float = 0.001
    gftt_min_distance: float = 20.0
    gftt_block_size: int = 3
    gftt_use_harris_detector: bool = False
    gftt_k: float = 0.04

    # --- sparse stereo params ---
    templ_rows: int = 11              # template height in pixels
    templ_cols: int = 101              # template width in pixels
    stripe_extra_rows: int = 0        # extra rows around template
    template_matching_tolerance: float = 0.15  # SSD (normalized) max allowed
    subpixel_refinement: bool = False
    stereo_min_depth: float = 0.15     # [m] min depth used to set max disparity

    # --- NEW: stereo RANSAC tuning ---
    stereo_ransac_min_inliers: int = 8
    stereo_ransac_confidence: float = 0.999


class KLTFeatureTracker:
    """
    KLT feature tracker that predicts depth via joint Left-Right tracking 
    and geometric verification using the frame's calibration Q matrix.
    """

    def __init__(
        self,
        *,
        max_feature_count: int = 1024,
        refill_feature_ratio: float = 0.8,
        feature_suppression_radius: float = 8.0,
        lk_win_size: tuple[int, int] = (15, 15),
        lk_max_level: int = 5,
        lk_max_iterations: int = 40,
        lk_epsilon: float = 0.01,
        lk_min_eig_threshold: float = 1e-3,
        max_depth: float = 40.0,

        gftt_quality_level: float = 0.001,
        gftt_min_distance: float = 20.0,
        gftt_block_size: int = 3,
        gftt_use_harris_detector: bool = False,
        gftt_k: float = 0.04,

        templ_rows: int = 11,
        templ_cols: int = 101,
        stripe_extra_rows: int = 0,
        template_matching_tolerance: float = 0.15,
        subpixel_refinement: bool = False,
        stereo_min_depth: float = 0.15,
        stereo_ransac_threshold: float = 2.0,
        stereo_max_y_diff: float = 2.0,
        min_disparity: float = 0.1,
    ) -> None:
        self.config = KLTTrackerConfig(
            max_feature_count=max_feature_count,
            refill_feature_ratio=refill_feature_ratio,
            feature_suppression_radius=feature_suppression_radius,
            lk_win_size=tuple(lk_win_size),
            lk_max_level=lk_max_level,
            lk_max_iterations=lk_max_iterations,
            lk_epsilon=lk_epsilon,
            lk_min_eig_threshold=lk_min_eig_threshold,
            max_depth=max_depth,

            gftt_quality_level=gftt_quality_level,
            gftt_min_distance=gftt_min_distance,
            gftt_block_size=gftt_block_size,
            gftt_use_harris_detector=gftt_use_harris_detector,
            gftt_k=gftt_k,
            templ_rows=templ_rows,
            templ_cols=templ_cols,
            stripe_extra_rows=stripe_extra_rows,
            template_matching_tolerance=template_matching_tolerance,
            subpixel_refinement=subpixel_refinement,
            stereo_min_depth=stereo_min_depth,
            stereo_ransac_threshold=stereo_ransac_threshold,
            stereo_max_y_diff=stereo_max_y_diff,
            min_disparity=min_disparity,
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

    def _detect_keypoints(
        self,
        gray_image: np.ndarray,
        max_features: int,
    ) -> np.ndarray:
        """Detect keypoints using GFTT (Good Features To Track)."""
        keypoints = cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=max_features,
            qualityLevel=self.config.gftt_quality_level,
            minDistance=self.config.gftt_min_distance,
            blockSize=self.config.gftt_block_size,
            useHarrisDetector=self.config.gftt_use_harris_detector,
            k=self.config.gftt_k,
        )

        return keypoints.reshape(-1, 2)

    def _reproject_sparse(
        self, 
        left_points: np.ndarray, 
        right_points: np.ndarray, 
        Q: np.ndarray,
        invert_disparity: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sparse equivalent of cv2.reprojectImageTo3D using the Q matrix.
        Includes options to flip disparity sign to handle Q matrix conventions.
        """
        # Calculate disparity (d = x_left - x_right)
        disparity = left_points[:, 0] - right_points[:, 0]
        
        if invert_disparity:
            disparity = -disparity

        # Prepare vectors for matrix multiplication: [x, y, d, 1]
        vectors = np.column_stack([
            left_points[:, 0],
            left_points[:, 1],
            disparity,
            np.ones_like(disparity)
        ])
        
        # Project: Result = V @ Q.T (shape: N, 4)
        points_homogenous = vectors @ Q.T
        
        # Normalize by W to get Cartesian coordinates
        W = points_homogenous[:, 3:4]
        
        # Avoid division by absolute zero, though huge coords will be filtered later
        W[np.abs(W) < 1e-6] = 1e-6 
        
        points3d = points_homogenous[:, :3] / W
        depths = points3d[:, 2] # Z coordinate
        
        return points3d, depths

    @profile
    def _search_right_epipolar(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        left_pt: np.ndarray,
        stripe_cols: int,
        stripe_rows: int,
    ) -> tuple[bool, np.ndarray, float]:
        """
        Kimera-style stereo search:
        - place a template around left_pt in the left image
        - search along a horizontal stripe in the right image using SSD template matching
        Returns (ok, match_pt, score), where score is the normalized SSD (lower is better).
        """
        cfg = self.config
        h_left, w_left = left_img.shape
        h_right, w_right = right_img.shape

        x, y = float(left_pt[0]), float(left_pt[1])
        rounded_x = int(round(x))
        rounded_y = int(round(y))

        templ_rows = int(cfg.templ_rows)
        templ_cols = int(cfg.templ_cols)

        # --- Place template in left image ---
        temp_corner_y = rounded_y - (templ_rows - 1) // 2
        if temp_corner_y < 0 or temp_corner_y + templ_rows > h_left:
            # Template would go out of vertical bounds
            return False, np.zeros(2, dtype=np.float32), -1.0

        temp_corner_x = rounded_x - (templ_cols - 1) // 2
        offset_temp = 0

        if temp_corner_x < 0:
            # Shift template right to stay inside image
            offset_temp = temp_corner_x
            temp_corner_x = 0

        if temp_corner_x + templ_cols > w_left:
            # Shift template left if it exceeds image on the right
            offset_temp = (temp_corner_x + templ_cols) - w_left
            temp_corner_x -= offset_temp
            if temp_corner_x < 0:
                # Cannot place template fully inside the image
                return False, np.zeros(2, dtype=np.float32), -1.0

        templ = left_img[
            temp_corner_y : temp_corner_y + templ_rows,
            temp_corner_x : temp_corner_x + templ_cols,
        ]
        if templ.shape[0] != templ_rows or templ.shape[1] != templ_cols:
            return False, np.zeros(2, dtype=np.float32), -1.0

        # --- Place stripe in right image ---
        stripe_corner_y = rounded_y - (stripe_rows - 1) // 2
        if stripe_corner_y < 0 or stripe_corner_y + stripe_rows > h_right:
            # Stripe would go out of vertical bounds
            return False, np.zeros(2, dtype=np.float32), -1.0

        stripe_corner_x = (
            rounded_x + (templ_cols - 1) // 2 - stripe_cols
        )  # left side of stripe
        offset_stripe = 0

        if stripe_corner_x + stripe_cols > w_right:
            # Stripe exceeds on the right; shift left
            offset_stripe = (stripe_corner_x + stripe_cols) - w_right
            stripe_corner_x -= offset_stripe

        if stripe_corner_x < 0:
            # Clamp to left border
            stripe_corner_x = 0

        stripe = right_img[
            stripe_corner_y : stripe_corner_y + stripe_rows,
            stripe_corner_x : stripe_corner_x + stripe_cols,
        ]
        if stripe.shape[0] < templ_rows or stripe.shape[1] < templ_cols:
            # Cannot run template matching if stripe smaller than template
            return False, np.zeros(2, dtype=np.float32), -1.0

        # --- Template matching (SSD) ---
        result = cv2.matchTemplate(stripe, templ, cv2.TM_SQDIFF)
        result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)

        # Map match location back to image coordinates
        match_x = (
            min_loc[0]
            + stripe_corner_x
            + (templ_cols - 1) // 2
            + offset_temp
        )
        match_y = (
            min_loc[1]
            + stripe_corner_y
            + (templ_rows - 1) // 2
        )
        match_pt = np.array([match_x, match_y], dtype=np.float32)

        # Subpixel refinement (optional)
        if self.config.subpixel_refinement:
            corner = match_pt.reshape(1, 1, 2).astype(np.float32)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                40,
                0.001,
            )
            cv2.cornerSubPix(right_img, corner, (10, 10), (-1, -1), criteria)
            match_pt = corner.reshape(2)

        return True, match_pt, float(min_val)

    @profile
    def _compute_stereo_matches(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        left_points: np.ndarray,
        Q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Use fast C++ Kimera-style sparse stereo to get right keypoints,
        then do Q-based reprojection and depth filtering in Python.
        """
        cfg = self.config
        n_points = int(left_points.shape[0])

        if n_points == 0:
            return (
                np.zeros(0, dtype=bool),
                np.empty((0, 2), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )

        # --- Stripe geometry (same as before) ---
        templ_rows = int(cfg.templ_rows)
        templ_cols = int(cfg.templ_cols)
        stripe_rows = templ_rows + int(cfg.stripe_extra_rows)

        fx = float(Q[2, 3])
        baseline = 1.0 / abs(Q[3, 2]) if abs(Q[3, 2]) > 1e-9 else 1.0
        min_depth = max(float(cfg.stereo_min_depth), 1e-3)
        stripe_cols = int(round(fx * baseline / min_depth) + templ_cols + 4)

        # --- Call C++ util (this is the heavy part) ---
        valid_mask_cpp, right_points_cpp, scores_cpp = stereo_matching.compute_sparse_stereo(
            left_gray,                        # HxW uint8
            right_gray,                       # HxW uint8
            left_points.astype(np.float32),   # Nx2 float32
            templ_rows,
            templ_cols,
            stripe_rows,
            stripe_cols,
            float(cfg.stereo_max_y_diff),
            float(cfg.min_disparity),
            float(cfg.template_matching_tolerance),
            bool(cfg.subpixel_refinement),
        )

        valid_mask_cpp = np.asarray(valid_mask_cpp, dtype=bool)
        right_points = np.asarray(right_points_cpp, dtype=np.float32)

        if not np.any(valid_mask_cpp):
            # No stereo matches
            return (
                np.zeros(n_points, dtype=bool),
                right_points,
                np.zeros(n_points, dtype=np.float32),
                np.zeros((n_points, 3), dtype=np.float32),
            )

        # --- Reproject with Q (your existing logic) ---
        valid_indices = np.where(valid_mask_cpp)[0]
        valid_left = left_points[valid_indices]
        valid_right = right_points[valid_indices]

        points3d_valid, depths_valid = self._reproject_sparse(
            valid_left, valid_right, Q, invert_disparity=False
        )

        # Flip sign if necessary
        finite_mask = np.isfinite(depths_valid)
        if np.any(finite_mask):
            pos = np.count_nonzero(depths_valid[finite_mask] > 0)
            neg = np.count_nonzero(depths_valid[finite_mask] < 0)
            if neg > pos:
                points3d_valid, depths_valid = self._reproject_sparse(
                    valid_left, valid_right, Q, invert_disparity=True
                )

        # Depth filtering
        depth_mask = (
            (depths_valid > 0.0)
            & (depths_valid <= cfg.max_depth)
            & np.isfinite(depths_valid)
        )

        final_valid_indices = valid_indices[depth_mask]

        final_mask = np.zeros(n_points, dtype=bool)
        full_depths = np.zeros(n_points, dtype=np.float32)
        full_points3d = np.zeros((n_points, 3), dtype=np.float32)

        if final_valid_indices.size > 0:
            final_mask[final_valid_indices] = True
            full_depths[final_valid_indices] = depths_valid[depth_mask].astype(np.float32)
            full_points3d[final_valid_indices] = points3d_valid[depth_mask].astype(np.float32)

        return final_mask, right_points, full_depths, full_points3d

    @profile
    def track_frame(
        self,
        rectified_frame: RectifiedStereoFrame,
    ) -> dict[int, TrackObservation]:
        """Track features for a single frame and update state."""
        frame_idx = len(self.track_history)
        cfg = self.config
        
        Q = rectified_frame.calibration.Q
        
        gray_left = cv2.cvtColor(rectified_frame.left_rect, cv2.COLOR_RGB2GRAY)
        gray_right = cv2.cvtColor(rectified_frame.right_rect, cv2.COLOR_RGB2GRAY)
        
        frame_obs: dict[int, TrackObservation] = {}
        tracked_points: list[np.ndarray] = []
        tracked_ids: list[int] = []

        # --- Step 1: Temporal Tracking (Left t-1 -> Left t) ---
        if self._prev_gray is not None and self._prev_points.size and self._prev_ids.size:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray,
                gray_left,
                self._prev_points,
                None,
                **self._lk_params,
            )
            
            if next_points is not None and status is not None:
                status_mask = status.reshape(-1).astype(bool)
                
                candidate_ids = self._prev_ids[status_mask]
                candidate_points = next_points.reshape(-1, 2)[status_mask]
                
                # --- Step 2: Stereo Tracking & Q-Matrix Verification ---
                valid_stereo, _, depths, _ = self._compute_stereo_matches(
                    gray_left, 
                    gray_right, 
                    candidate_points,
                    Q
                )
                
                final_ids = candidate_ids[valid_stereo]
                final_points = candidate_points[valid_stereo]
                final_depths = depths[valid_stereo]
                
                for i, track_id in enumerate(final_ids):
                    point = final_points[i]
                    depth_val = final_depths[i]
                    
                    track = self.tracks[track_id]
                    track.add_observation(frame_idx, point, depth_val)
                    frame_obs[track_id] = track.observations[frame_idx]
                    
                    tracked_points.append(point)
                    tracked_ids.append(track_id)
            
            active_set = set(tracked_ids)
            for track_id in self._prev_ids:
                if track_id not in active_set:
                    self.tracks[track_id].active = False

        if tracked_points:
            existing_points = np.asarray(tracked_points, dtype=np.float32)
        else:
            existing_points = np.empty((0, 2), dtype=np.float32)

        # --- Step 3: Refill with New Features ---
        def add_new_tracks(budget: int) -> tuple[list[int], np.ndarray]:
            nonlocal existing_points
            if budget <= 0:
                return [], np.empty((0, 2), dtype=np.float32)

            detection_quota = max(cfg.max_feature_count, budget * 2)
            candidate_points = self._detect_keypoints(gray_left, detection_quota)
            
            filtered_candidates = self._filter_keypoints_by_distance(
                candidate_points,
                existing_points,
                cfg.feature_suppression_radius,
            )
            
            if filtered_candidates.size == 0:
                return [], np.empty((0, 2), dtype=np.float32)

            # Stereo Match & Q-Matrix Reprojection
            valid_mask, _, depths, points3d = self._compute_stereo_matches(
                gray_left,
                gray_right,
                filtered_candidates,
                Q
            )
            
            valid_candidates = filtered_candidates[valid_mask]
            valid_depths = depths[valid_mask]
            valid_points3d = points3d[valid_mask]
            
            take = min(budget, valid_candidates.shape[0])
            new_ids: list[int] = []
            new_points_list: list[np.ndarray] = []
            
            h, w, _ = rectified_frame.left_rect.shape
            
            for i in range(take):
                keypoint = valid_candidates[i]
                depth_val = float(valid_depths[i])
                point3 = valid_points3d[i].astype(np.float64)
                
                ix, iy = int(keypoint[0]), int(keypoint[1])
                color = np.array([0, 0, 0], dtype=np.uint8)
                if 0 <= iy < h and 0 <= ix < w:
                    color = rectified_frame.left_rect[iy, ix]

                track = FeatureTrack(
                    track_id=self._next_track_id,
                    anchor_frame=frame_idx,
                    anchor_keypoint=keypoint.astype(np.float32),
                    anchor_depth=depth_val,
                    anchor_point3=point3,
                    anchor_color=color,
                )
                track.add_observation(frame_idx, keypoint, depth_val)
                self.tracks[self._next_track_id] = track
                
                new_ids.append(self._next_track_id)
                new_points_list.append(keypoint.astype(np.float32))
                frame_obs[self._next_track_id] = track.observations[frame_idx]
                self._next_track_id += 1

            if new_points_list:
                new_points_arr = np.asarray(new_points_list, dtype=np.float32)
                if existing_points.size:
                    existing_points = np.vstack([existing_points, new_points_arr])
                else:
                    existing_points = new_points_arr
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
        self._prev_gray = gray_left
        self._prev_points = stacked_points
        self._prev_ids = active_ids
        return frame_obs

    def track(
        self,
        rectified_frames: list[RectifiedStereoFrame],
    ) -> tuple[list[dict[int, TrackObservation]], dict[int, FeatureTrack]]:
        self.reset()
        for rectified_frame in rectified_frames:
            self.track_frame(rectified_frame)
        return list(self.track_history), dict(self.tracks)