"""Bundle adjustment utilities for the local VO pipeline."""

from typing import Literal

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, L, V, X

from slam.registration.registration import RectifiedStereoFrame

from .klt_tracker import TrackObservation, TrackObservationsBatch


@dataclass
class BundleAdjustmentConfig:
    """Configuration container for bundle adjustment."""

    optimizer: Literal["incremental", "batch"] = "incremental"
    min_observations_per_landmark: int = 3
    min_observations_per_frame: int = 5
    max_observations_per_landmark: int = 3
    max_landmarks: int = 10_000
    projection_noise_px: float = 1.0
    disparity_noise_px: float = 1.0
    use_huber_loss: bool = True
    reprojection_gating: bool = True
    reprojection_gating_threshold_px: float = 2.0
    max_rejection_rate: float = 1.01
    use_median_filtering: bool = False
    reprojection_gating_min_obs: int = 12
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
    use_motion_only_smart_factors: bool = False
    use_light_relative_pose_factor: bool = False
    light_relative_pose_sigmas: Sequence[float] = field(
        default_factory=lambda: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    )
    min_depth: float = 0.2
    lag: float = 3.0
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


# remember in developing this: the methods should be simple, if anything gets long or complicated, factor it out into composable components that ARE simple
class FixedLagBundleAdjuster:
    def __init__(self, config: BundleAdjustmentConfig) -> None:
        self.config = config
        
        self.frame_idx = 0
        self.prev_ts = 0.0

        self.ts = []

        self.smoother = None
        self.smoothed_values = None
        self.full_values = None
        
        self.landmark_observations = {}
        self.landmark_key_map = {}
        self.next_replacement_id = 1000000

        self.new_factors = None
        self.new_values = None
        self.new_timestamps = None

    def reset(self, ts: float, pose: gtsam.Pose3, velocity: np.ndarray, bias: gtsam.imuBias.ConstantBias) -> None:
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()
        self.new_timestamps = {}
        
        if self.config.optimizer == "incremental":
            self.smoother = gtsam.IncrementalFixedLagSmoother(self.config.lag)
        elif self.config.optimizer == "batch":
            self.smoother = gtsam.BatchFixedLagSmoother(self.config.lag)
        
        self.smoothed_values = gtsam.Values()
        self.full_values = gtsam.Values()
        self.landmark_observations = {}
        self.landmark_key_map = {}
        self.next_replacement_id = 1000000

        self.frame_idx = 0
        self.prev_ts = ts

        self.ts.append(ts)

        self._add_prior_factors(ts, pose, velocity, bias)

    def process(self, frame: RectifiedStereoFrame, ts: float, relative_pose: gtsam.Pose3, estimated_velocity: np.ndarray, landmark_observations: Mapping[int, TrackObservation] | TrackObservationsBatch, pim: gtsam.PreintegratedCombinedMeasurements, optimize: bool = True, profile: bool = False) -> tuple[dict[str, int], list[str]]:
        profiler = _SectionProfiler(log_prefix="fixed_lag_bundle_adjustment.process")

        stats: dict[str, int] = {
            "frame_idx": self.frame_idx + 1,
            "selected": 0,
            "features_added": 0,
            "rejected_depth": 0,
            "rejected_reprojection": 0,
            "rejected_not_selected": 0,
            "active_landmarks": 0,
        }
        warnings = []

        if not isinstance(landmark_observations, TrackObservationsBatch):
            landmark_observations = TrackObservationsBatch.from_any(landmark_observations)

        # Extract Extrinsics (Body -> Sensor)
        body_P_sensor = frame.calibration.imu_from_left
        
        # Predict Initial Guess
        prev_pose = self.full_values.atPose3(X(self.frame_idx))
        current_pose_guess = prev_pose.compose(relative_pose)
        
        prev_idx = self.frame_idx
        self.frame_idx += 1

        # 1. Add Values & IMU
        with profiler.section("add_values_imu"):
            self._insert_initial_values(self.frame_idx, ts, current_pose_guess, estimated_velocity, prev_idx)
            # Note: relying on internal strict params, ignoring passed 'pim_unused'
            # You must update your main loop to pass raw IMU data or generate PIM with the same strict params!
            self._add_imu_factor(prev_idx, self.frame_idx, pim, ts)

        # add light relative pose factor to keep the drift from being way too large
        # use a L2WithDeadZone loss to so that it doesn't have much of an effect unless the pose is way off
        if self.config.use_light_relative_pose_factor:
            self._add_light_relative_pose_factor(prev_idx, self.frame_idx, ts)

        # 2. Process Landmarks
        with profiler.section("process_landmarks"):
            self._process_explicit_landmarks(frame, landmark_observations, body_P_sensor, ts, stats=stats, warnings=warnings)

        # 3. Optimize
        if optimize:
            with profiler.section("optimize"):
                self.optimize()

        self.prev_ts = ts

        self.ts.append(ts)

        stats["active_landmarks"] = self._count_active_landmarks()

        return stats, warnings

    def optimize(self) -> None:
        self._prune_unconstrained_values()
        try:
            self.smoother.update(self.new_factors, self.new_values, self.new_timestamps)
            self.smoothed_values = self.smoother.calculateEstimate()
            self.full_values.update(self.smoothed_values)
        except Exception as e:
            print(f"Optimization Error at frame {self.frame_idx}: {e}")
            # Debug: Print graph stats or active keys if needed
            raise e
        finally:
            self.new_timestamps.clear()
            self.new_values.clear()
            self.new_factors.resize(0)

    def get_trajectory(self) -> list[gtsam.Pose3]:
        return [self.full_values.atPose3(X(i)) for i in range(self.frame_idx + 1)]

    def get_trajectory_velocities(self) -> list[np.ndarray]:
        return [self.full_values.atVector(V(i)) for i in range(self.frame_idx + 1)]

    def get_latest_pose(self) -> gtsam.Pose3:
        return self.full_values.atPose3(X(self.frame_idx))

    def get_latest_velocity(self) -> np.ndarray:
        return self.full_values.atVector(V(self.frame_idx))
        # return np.zeros(3) # remember to disable this lol

    def get_bias(self) -> gtsam.imuBias.ConstantBias:
        return self.full_values.atConstantBias(B(self.frame_idx))
        # return gtsam.imuBias.ConstantBias() # and this too lmao

    def get_trajectory_biases(self, ret: Literal["list", "numpy"] = "list") -> list[gtsam.imuBias.ConstantBias] | np.ndarray:
        if ret == "list":
            return [self.full_values.atConstantBias(B(i)) for i in range(self.frame_idx + 1)]
        elif ret == "numpy":
            return np.array([
                np.concatenate([
                    [ts],
                    bias.accelerometer(),
                    bias.gyroscope(),
                ])
                for ts, bias in zip(self.ts, self.get_trajectory_biases("list"))
            ])
        else:
            raise ValueError(f"Unknown type: {type}")

    def get_trajectory_timestamps(self) -> list[float]:
        return self.ts

    def get_active_landmarks(self) -> list[dict[str, np.ndarray | int]]:
        """Return landmark positions inside the current smoothing window.

        Each entry includes the active landmark id (which may differ from the
        original track id after a respawn), the original track id, and the
        3D position in world coordinates.
        """
        active: dict[int, np.ndarray] = {}

        def _collect(values: gtsam.Values | None) -> None:
            if values is None:
                return
            for key in values.keys():
                symbol = gtsam.Symbol(key)
                if symbol.chr() != ord("l"):
                    continue
                active[symbol.index()] = values.atPoint3(key)

        _collect(self.smoothed_values)
        _collect(self.new_values)

        if not active:
            return []

        inverse_map = {current_id: original_id for original_id, current_id in self.landmark_key_map.items()}
        landmarks = []
        for landmark_id, position in active.items():
            original_id = inverse_map.get(landmark_id, landmark_id)
            landmarks.append(
                {
                    "landmark_id": int(landmark_id),
                    "original_track_id": int(original_id),
                    "position": position,
                }
            )

        # sort by original id for deterministic ordering
        return sorted(landmarks, key=lambda entry: entry["original_track_id"])

    def get_all_landmarks(self) -> list[dict[str, np.ndarray | int]]:
        """Return ALL landmarks from the full history.

        Each entry includes the active landmark id (which may differ from the
        original track id after a respawn), the original track id, and the
        3D position in world coordinates.
        """
        active: dict[int, np.ndarray] = {}

        def _collect(values: gtsam.Values | None) -> None:
            if values is None:
                return
            for key in values.keys():
                symbol = gtsam.Symbol(key)
                if symbol.chr() != ord("l"):
                    continue
                active[symbol.index()] = values.atPoint3(key)

        _collect(self.full_values)

        if not active:
            return []

        inverse_map = {current_id: original_id for original_id, current_id in self.landmark_key_map.items()}
        landmarks = []
        for landmark_id, position in active.items():
            original_id = inverse_map.get(landmark_id, landmark_id)
            landmarks.append(
                {
                    "landmark_id": int(landmark_id),
                    "original_track_id": int(original_id),
                    "position": position,
                }
            )

        # sort by original id for deterministic ordering
        return sorted(landmarks, key=lambda entry: entry["original_track_id"])

    # =========================================================================
    # Helpers
    # =========================================================================

    def _add_light_relative_pose_factor(self, prev_idx: int, idx: int, ts: float) -> None:
        """
        Add a light relative pose factor to keep the drift from being way too large.
        
        Uses a L2WithDeadZone loss function to so that it doesn't have much of an effect unless the pose is way off.
        """
        prev_pose = self.full_values.atPose3(X(prev_idx))
        current_pose = self.full_values.atPose3(X(idx))
        light_relative_pose = prev_pose.inverse().compose(current_pose)

        noise = gtsam.noiseModel.Robust(
            gtsam.noiseModel.mEstimator.L2WithDeadZone.Create(1.0),
            gtsam.noiseModel.Diagonal.Sigmas(self.config.light_relative_pose_sigmas)
        )

        self.new_factors.add(gtsam.PriorFactorPose3(X(idx), light_relative_pose, noise))

    def _count_active_landmarks(self) -> int:
        """Count landmarks that are still inside the fixed-lag smoother window."""
        active_keys: set[int] = set()
        if self.smoothed_values is not None:
            for key in self.smoothed_values.keys():
                if gtsam.Symbol(key).chr() == ord("l"):
                    active_keys.add(key)
        if self.new_values is not None:
            for key in self.new_values.keys():
                if gtsam.Symbol(key).chr() == ord("l"):
                    active_keys.add(key)
        return len(active_keys)

    def _process_explicit_landmarks(self, frame: RectifiedStereoFrame, observations, body_P_sensor, ts, stats: dict[str, int] | None = None, warnings: list[str] | None = None):
        # Prepare Calibration / Noise
        K = frame.calibration.K_left_rect
        calib = gtsam.Cal3_S2(float(K[0,0]), float(K[1,1]), float(K[0,1]), float(K[0,2]), float(K[1,2]))
        
        stereo_calib = gtsam.Cal3_S2Stereo(
            float(K[0,0]), float(K[1,1]), float(K[0,1]), float(K[0,2]), float(K[1,2]),
            float(np.linalg.norm(frame.calibration.T)) # TODO: is this way of calculating the baseline the most accurate? should it be factor out to a more central place, there's a few places where we calculate it and having one function might be good. compose don't inherit.
        )
        
        # # Select the best tracks
        # initial_observation_count = len(observations)
        # observations = self._select_best_tracks(observations)
        # if stats is not None:
        #     stats["selected"] = len(observations)
        #     stats["rejected_not_selected"] += max(initial_observation_count - len(observations), 0)
        
        # Stereo Noise
        sigma = self.config.projection_noise_px
        stereo_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber(1.0), # TODO: turn K into a configurable parameter
            # gtsam.noiseModel.mEstimator.Cauchy(1.345),
            gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma, sigma, self.config.disparity_noise_px]))
        )

        # Helper to get 3D point in World Frame
        # NOTE: We perform backprojection using the CURRENT estimated pose (Body) + Extrinsics
        pose_body = self.new_values.atPose3(X(self.frame_idx))
        pose_cam = pose_body.compose(body_P_sensor)
        camera = gtsam.PinholeCameraCal3_S2(pose_cam, calib)

        if isinstance(observations, TrackObservationsBatch):
            ids_arr = observations.ids.astype(int)
            kps_arr = observations.keypoints
            depths_arr = observations.depths
            obs_iter = list(zip(ids_arr.tolist(), kps_arr, depths_arr))
        else:
            obs_iter = [
                (tid, obs.keypoint, obs.depth) for tid, obs in observations.items()
            ]

        # Collection for existing landmark factors to potentially filter
        # List of dicts: {factor, error, passed_threshold, is_new=False}
        candidate_factors = []

        for original_track_id, keypoint, depth_val in obs_iter:
            if depth_val is None or not np.isfinite(depth_val) or depth_val <= 0.0 or depth_val > 40.0:
                if stats is not None:
                    stats["rejected_depth"] += 1
                continue

            if original_track_id not in self.landmark_key_map:
                self.landmark_key_map[original_track_id] = original_track_id
            track_id = self.landmark_key_map[original_track_id]
            key_L = L(track_id)

            # Create Factor
            disparity = (stereo_calib.fx() * stereo_calib.baseline()) / depth_val
            stereo_meas = gtsam.StereoPoint2(
                float(keypoint[0]),
                float(keypoint[0] - disparity),
                float(keypoint[1])
            )

            # FIX 2: Use GenericStereoFactor3D with BODY POSE + EXTRINSICS
            # This allows GTSAM to optimize the Body path correctly
            factor = gtsam.GenericStereoFactor3D(
                stereo_meas, stereo_noise, 
                X(self.frame_idx), key_L, 
                stereo_calib, body_P_sensor
            )

            # Landmark Logic
            if self.full_values.exists(key_L):
                # Calculate error
                error = 0.0
                passed_threshold = True
                
                existing_point = self.full_values.atPoint3(key_L)
                try:
                    projected_pt2 = camera.project(existing_point)
                    residual = projected_pt2 - gtsam.Point2(float(keypoint[0]), float(keypoint[1]))
                    error = np.linalg.norm(residual)
                except RuntimeError:
                    # Projection failed (behind camera, etc.)
                    # We treat this as infinite error / fail
                    error = float('inf')
                    passed_threshold = False
                
                if self.config.reprojection_gating and error > self.config.reprojection_gating_threshold_px:
                    passed_threshold = False

                # Landmark exists in history.
                # Check if it is currently optimized (alive in the smoother window)
                if self.smoothed_values.exists(key_L) or self.new_values.exists(key_L):
                    # It's alive. Add to candidates.
                    candidate_factors.append({
                        "factor": factor,
                        "error": error,
                        "passed_threshold": passed_threshold,
                    })
                else:
                    # It was marginalized out (exists in full_values but not smoothed).
                    # Strategy: Respawn as NEW landmark ID
                    self.next_replacement_id += 1
                    self.landmark_key_map[original_track_id] = self.next_replacement_id
                    new_track_id = self.next_replacement_id
                    key_L_new = L(new_track_id)
                    
                    # Re-triangulate fresh point
                    point = camera.backproject(keypoint, depth_val)
                    
                    self.new_values.insert(key_L_new, point)
                    self.full_values.insert(key_L_new, point)
                    self.new_timestamps[key_L_new] = ts
                    
                    # Create new factor for new ID
                    new_factor = gtsam.GenericStereoFactor3D(
                        stereo_meas, stereo_noise, 
                        X(self.frame_idx), key_L_new, 
                        stereo_calib, body_P_sensor
                    )
                    
                    self.new_factors.add(new_factor)
                    if stats is not None:
                        stats["features_added"] += 1

                    # add a weak prior on the landmark to stop it from drifting too far
                    prior_factor = gtsam.PriorFactorPoint3(key_L_new, point, gtsam.noiseModel.Diagonal.Sigmas(np.array([1e+2]*3)))
                    self.new_factors.add(prior_factor)

            else:
                # New Landmark
                point = camera.backproject(keypoint, depth_val)
                
                self.new_values.insert(key_L, point)
                self.full_values.insert(key_L, point)
                self.new_timestamps[key_L] = ts
                self.new_factors.add(factor)
                if stats is not None:
                    stats["features_added"] += 1

                # add a weak prior on the landmark to stop it from drifting too far
                prior_factor = gtsam.PriorFactorPoint3(key_L, point, gtsam.noiseModel.Diagonal.Sigmas(np.array([1e+2]*3)))
                self.new_factors.add(prior_factor)

        # --- Filtering Logic ---
        
        # 1. Determine Gating Policy (Safety Valve)
        disable_gating = False
        if self.config.reprojection_gating and candidate_factors:
            rejected_count = sum(1 for c in candidate_factors if not c["passed_threshold"])
            rejection_rate = rejected_count / len(candidate_factors)
            
            if rejection_rate >= self.config.max_rejection_rate:
                disable_gating = True
                # if stats is not None:
                #     stats["gating_disabled"] = 1

        # 2. Select Survivors
        survivors = []
        rejected_by_gating = 0
        
        if candidate_factors:
            for c in candidate_factors:
                if self.config.reprojection_gating and not disable_gating:
                    if not c["passed_threshold"]:
                        rejected_by_gating += 1
                        continue
                survivors.append(c)

        # 3. Apply Median Filtering
        rejected_by_median = 0
        if self.config.use_median_filtering and survivors:
            errors = np.array([c["error"] for c in survivors])
            median_error = np.median(errors)
            
            # Filter
            filtered_survivors = [c for c in survivors if c["error"] <= median_error]
            rejected_by_median = len(survivors) - len(filtered_survivors)
            survivors = filtered_survivors

        # 4. Min Obs Rescue
        factors_to_add = survivors
        rescued = False
        
        if len(survivors) < self.config.reprojection_gating_min_obs:
             # Sort ALL candidates by error (asc)
             sorted_candidates = sorted(candidate_factors, key=lambda c: c["error"])
             # Take best N
             factors_to_add = sorted_candidates[:self.config.reprojection_gating_min_obs]
             rescued = True
             
             warnings.append(f"Low observation count, try decreasing speed or improve lighting conditions for better tracking")
        
        # 5. Add to Graph & Update Stats
        for item in factors_to_add:
            self.new_factors.add(item["factor"])
            if stats is not None:
                stats["features_added"] += 1
        
        if stats is not None:
            if not rescued:
                stats["rejected_reprojection"] += rejected_by_gating + rejected_by_median
            else:
                 # If rescued, true rejections is just total - added
                 stats["rejected_reprojection"] += max(0, len(candidate_factors) - len(factors_to_add))


    def _select_best_tracks(self, observations, limit=1024): # TODO: this limit should be part of the hydra config
        """
        Selects the top 'limit' tracks to add to the optimizer.
        Heuristic: Prioritize oldest tracks (smaller track_id).
        """
        if len(observations) <= limit:
            return observations

        # Sort by track_id (assuming strictly increasing IDs, smaller = older)
        # In a real KLT, older tracks provide stronger drift constraints.
        sorted_ids = sorted(observations.keys())
        
        # Simple top-N
        # (A better version would ensure spatial grid distribution, but this is fast)
        selected_ids = sorted_ids[:limit]
        
        return {tid: observations[tid] for tid in selected_ids}



    def _add_prior_factors(self, ts, pose, velocity, bias):
        key_pose = X(0)
        self.new_values.insert(key_pose, pose)
        self.full_values.insert(key_pose, pose)
        self.new_timestamps[key_pose] = ts
        self.new_factors.add(gtsam.PriorFactorPose3(key_pose, pose, gtsam.noiseModel.Diagonal.Sigmas(np.asarray(self.config.pose_prior_sigmas))))

        key_vel = V(0)
        vel_pt = gtsam.Point3(*velocity)
        self.new_values.insert(key_vel, vel_pt)
        self.full_values.insert(key_vel, vel_pt)
        self.new_timestamps[key_vel] = ts
        
        key_bias = B(0)
        self.new_values.insert(key_bias, bias)
        self.full_values.insert(key_bias, bias)
        self.new_timestamps[key_bias] = ts

        # Tighter initial bias prior
        bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(self.config.imu_bias_prior_for_factors))
        self.new_factors.add(gtsam.PriorFactorConstantBias(key_bias, bias, bias_noise))

    def _insert_initial_values(self, curr_idx, ts, pose, velocity, prev_idx):
        self.new_values.insert(X(curr_idx), pose)
        self.full_values.insert(X(curr_idx), pose)
        self.new_timestamps[X(curr_idx)] = ts

      
        # TODO: how does not doing this work? isn't this velocity incorrectly in the body frame instead of the world frame?
        # # transform the body velocity into the world frame
        # world_velocity = pose.inverse().rotation().rotate(velocity)
        self.new_values.insert(V(curr_idx), velocity)
        self.full_values.insert(V(curr_idx), velocity)
        self.new_timestamps[V(curr_idx)] = ts

        bias = self.full_values.atConstantBias(B(prev_idx))
        self.new_values.insert(B(curr_idx), bias)
        self.full_values.insert(B(curr_idx), bias)
        self.new_timestamps[B(curr_idx)] = ts

    def _add_imu_factor(self, prev_idx, curr_idx, pim, ts):
        self.new_factors.add(
            gtsam.CombinedImuFactor(
                X(prev_idx), V(prev_idx),
                X(curr_idx), V(curr_idx),
                B(prev_idx),
                B(curr_idx),
                pim
            )
        )
        # pass

    def _prune_unconstrained_values(self):
        """
        Checks new_values for variables that are NOT connected to any factor in new_factors.
        This prevents 'Indeterminant linear system' errors in BatchFixedLagSmoother.
        """
        # 1. Identify all keys referenced by new factors
        referenced_keys = set()
        for i in range(self.new_factors.size()):
            factor = self.new_factors.at(i)
            for key in factor.keys():
                referenced_keys.add(key)
        
        # 2. Check new values
        keys_to_remove = []
        # Iterate over keys in new_values
        for key in self.new_values.keys():
            # Only prune Landmarks (L) to be safe, though technically any unconstrained var is bad
            if gtsam.Symbol(key).chr() == ord('l'):
                if key not in referenced_keys:
                    keys_to_remove.append(key)
        
        # 3. Remove them
        for key in keys_to_remove:
            self.new_values.erase(key)
            if key in self.new_timestamps:
                del self.new_timestamps[key]
            # Also remove from full_values so we don't think it exists later
            if self.full_values.exists(key):
                # GTSAM Values doesn't have erase for full_values easily without rebuilding,
                # but full_values is just a log. 
                # The critical part is removing from new_values before smoother.update()
                pass

def finite_difference_velocity(prev_pose: gtsam.Pose3, next_pose: gtsam.Pose3, dt: float) -> gtsam.Point3:
    """Calculates the velocity between two poses using finite difference."""
    return (next_pose.translation() - prev_pose.translation()) / dt
