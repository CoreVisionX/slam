from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import gtsam
import numpy as np
from gtsam.symbol_shorthand import X

from slam.registration.registration import RectifiedStereoFrame
from slam.vio import stereo_matching

@dataclass
class OrbRelocalizationConfig:
    """Configuration for ORB Relocalization."""
    
    enabled: bool = False
    
    # ORB Parameters
    n_features: int = 500
    scale_factor: float = 1.2
    n_levels: int = 8
    edge_threshold: int = 31
    first_level: int = 0
    wta_k: int = 2
    score_type: int = cv2.ORB_HARRIS_SCORE
    patch_size: int = 31
    fast_threshold: int = 20
    
    # Matching Parameters
    match_ratio_thresh: float = 0.75
    
    # PnP Parameters
    pnp_reprojection_error: float = 2.0
    pnp_confidence: float = 0.999
    pnp_iterations: int = 100
    min_inliers: int = 30
    pnp_method: int = cv2.SOLVEPNP_EPNP
    
    # Execution parameters
    run_every_frames: int = 10
    
    # Factor noise
    prior_rotation_sigma: float = 0.05  # rad
    prior_translation_sigma: float = 0.02  # meters

    # Sparse Stereo Parameters (defaults from KLT)
    templ_rows: int = 11
    templ_cols: int = 101
    stripe_extra_rows: int = 0
    template_matching_tolerance: float = 0.15
    subpixel_refinement: bool = False
    stereo_min_depth: float = 0.15
    stereo_max_y_diff: float = 2.0
    min_disparity: float = 0.1


class OrbRelocalization:
    def __init__(self, config: OrbRelocalizationConfig):
        self.config = config
        
        self.origin_pose: Optional[gtsam.Pose3] = None
        self.origin_descriptors: Optional[np.ndarray] = None
        self.origin_keypoints: list[cv2.KeyPoint] = []
        self.origin_points_3d: Optional[np.ndarray] = None
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=self.config.n_features,
            scaleFactor=self.config.scale_factor,
            nlevels=self.config.n_levels,
            edgeThreshold=self.config.edge_threshold,
            firstLevel=self.config.first_level,
            WTA_K=self.config.wta_k,
            scoreType=self.config.score_type,
            patchSize=self.config.patch_size,
            fastThreshold=self.config.fast_threshold,
        )
        
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

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
        
        # Avoid division by absolute zero
        W[np.abs(W) < 1e-6] = 1e-6 
        
        points3d = points_homogenous[:, :3] / W
        depths = points3d[:, 2] # Z coordinate
        
        return points3d, depths

    def reset(self, frame: RectifiedStereoFrame, pose: gtsam.Pose3) -> None:
        """Initialize the relocalization module with the first frame (origin)."""
        if not self.config.enabled:
            return
            
        self.origin_pose = pose
        
        # Detect ORB features on Left Image
        left_img = frame.left_rect
        right_img = frame.right_rect
        
        if left_img.ndim == 3:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        if right_img.ndim == 3:
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
        kp, des = self.orb.detectAndCompute(left_img, None)
        
        if kp is None or des is None or len(kp) == 0:
            print("Warning: No ORB features found in origin frame for relocalization.")
            return

        # Prepare points for sparse stereo
        left_points = np.array([k.pt for k in kp], dtype=np.float32)
        
        # Compute Sparse Stereo Config
        Q = frame.calibration.Q
        fx = float(Q[2, 3])
        baseline = 1.0 / abs(Q[3, 2]) if abs(Q[3, 2]) > 1e-9 else 1.0
        min_depth = max(float(self.config.stereo_min_depth), 1e-3)
        stripe_cols = int(round(fx * baseline / min_depth) + self.config.templ_cols + 4)
        stripe_rows = self.config.templ_rows + int(self.config.stripe_extra_rows)

        # Call C++ Sparse Stereo
        try:
             valid_mask_cpp, right_points_cpp, scores_cpp = stereo_matching.compute_sparse_stereo(
                left_img,
                right_img,
                left_points,
                self.config.templ_rows,
                self.config.templ_cols,
                stripe_rows,
                stripe_cols,
                float(self.config.stereo_max_y_diff),
                float(self.config.min_disparity),
                float(self.config.template_matching_tolerance),
                bool(self.config.subpixel_refinement),
            )
        except Exception as e:
            print(f"Error in sparse stereo matching: {e}")
            return

        valid_mask = np.asarray(valid_mask_cpp, dtype=bool)
        
        if not np.any(valid_mask):
            print("Warning: No valid stereo matches for ORB features.")
            return
            
        right_points = np.asarray(right_points_cpp, dtype=np.float32)
        
        # Filter valid
        valid_indices = np.where(valid_mask)[0]
        valid_left = left_points[valid_indices]
        valid_right = right_points[valid_indices]
        valid_des = des[valid_indices]
        valid_kp_objs = [kp[i] for i in valid_indices]
        
        # Reproject to 3D
        points3d, depths = self._reproject_sparse(valid_left, valid_right, Q, invert_disparity=False)
        
        # Check standard convention (depth > 0)
        # If Q matrix conventions result in negative depths, flip
        pos_depths = np.count_nonzero(depths > 0)
        neg_depths = np.count_nonzero(depths < 0)
        if neg_depths > pos_depths:
            points3d, depths = self._reproject_sparse(valid_left, valid_right, Q, invert_disparity=True)
            
        # Final filtering for valid depth
        mask_depth = (depths > 0) & np.isfinite(depths)
        
        if not np.any(mask_depth):
            print("Warning: No features with valid depth found.")
            return
            
        self.origin_keypoints = [valid_kp_objs[i] for i in range(len(valid_kp_objs)) if mask_depth[i]]
        self.origin_descriptors = valid_des[mask_depth]
        self.origin_points_3d = points3d[mask_depth]
        
        print(f"[Relocalization] Initialized with {len(self.origin_points_3d)} features.")

    def process(self, frame_idx: int, symbol_index: int, frame: RectifiedStereoFrame) -> Optional[gtsam.NonlinearFactor]:
        """
        Process a frame and attempt to find a loop closure to the origin.
        Result is a PriorFactorPose3 if successful.
        """
        if not self.config.enabled:
            return None
            
        if self.origin_descriptors is None or self.origin_points_3d is None:
            return None
            
        if frame_idx % self.config.run_every_frames != 0:
            return None
            
        # Detect ORB on current frame
        left_img = frame.left_rect
        if left_img.ndim == 3:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            
        kp, des = self.orb.detectAndCompute(left_img, None)
        
        if des is None or len(kp) < self.config.min_inliers:
            return None
            
        # Match against origin
        matches = self.bf_matcher.knnMatch(des, self.origin_descriptors, k=2)
        
        # Ratio Test
        good_matches = []
        for m, n in matches:
            if m.distance < self.config.match_ratio_thresh * n.distance:
                good_matches.append(m)
                
        if len(good_matches) < self.config.min_inliers:
            return None
            
        # Prepare for PnP
        # queryIdx is current frame (kp), trainIdx is origin frame (3d points)
        obj_points = []
        img_points = []
        
        for m in good_matches:
            obj_points.append(self.origin_points_3d[m.trainIdx])
            img_points.append(kp[m.queryIdx].pt)
            
        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)
        
        # Solve PnP
        K = frame.calibration.K_left_rect
        camera_matrix = np.array([
            [K[0,0], K[0,1], K[0,2]],
            [K[1,0], K[1,1], K[1,2]],
            [K[2,0], K[2,1], K[2,2]]
        ], dtype=np.float64)
        dist_coeffs = np.zeros(4, dtype=np.float64) # Images are rectified
        
        # To make it robust, use RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            iterationsCount=self.config.pnp_iterations,
            reprojectionError=self.config.pnp_reprojection_error,
            confidence=self.config.pnp_confidence,
            flags=self.config.pnp_method
        )
        
        if not success or inliers is None or len(inliers) < self.config.min_inliers:
            return None
            
        # PnP Result: T_current_from_origin
        # We need Pose_current = Pose_origin * T_current_from_origin.inverse()
        
        R_current_from_origin, _ = cv2.Rodrigues(rvec)
        t_current_from_origin = tvec.flatten()
        
        # Create GTSAM pose
        T_current_from_origin = gtsam.Pose3(
            gtsam.Rot3(R_current_from_origin),
            gtsam.Point3(*t_current_from_origin)
        )
        
        T_origin_from_current = T_current_from_origin.inverse()
        
        Pose_current = self.origin_pose.compose(T_origin_from_current)
        
        # Create PriorFactor
        # Noise model
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
            self.config.prior_rotation_sigma, self.config.prior_rotation_sigma, self.config.prior_rotation_sigma,
            self.config.prior_translation_sigma, self.config.prior_translation_sigma, self.config.prior_translation_sigma
        ]))
        
        return gtsam.PriorFactorPose3(X(symbol_index), Pose_current, noise)
