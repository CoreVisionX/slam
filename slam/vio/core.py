from pathlib import Path
from typing import Sequence

import gtsam
import numpy as np
from hydra.utils import instantiate
from line_profiler import profile

from slam.depth.sgbm import SGBM
from slam.hydra_utils import compose_config, extract_target_config
from slam.registration.registration import RectifiedStereoFrame, StereoDepthFrame
from slam.vio.bundle_adjustment import FixedLagBundleAdjuster, finite_difference_velocity
from slam.vio.klt_tracker import KLTFeatureTracker
from slam.vio.klt_tracker_cpp import KLTFeatureTrackerCpp
from slam.vio.relative_pose import RelativePnPInitializer
from .config import VIOConfig, compute_vio_calibration
from .io import VIORerunLogger
from .imu_preintegration import ImuPreintegrator
from .types import VIOEstimate

# TODO: see if switching to CombinedImuFactors helps on longer sequences by accounting for IMU bias drift
# TODO: see if relative pose initialization via Imu Preintegration is better than PnP

# support for absolute orientation with magnometers (or even GPS) would be super useful and not that hard to add

# TODO: see if adding keyframing helps accuracy and performance at all. definitely could by making longer lag windows much more feasible
# TODO: see if adding a minimum number of observations per landmark before it's added to the graph helps accuracy

# TODO: offer a way to get poses at the IMU update rate
# maybe that should come with multithreading support?
# TODO: timing/profiling capabilities. should be well integrated with VIORerunLogger for easy debugging
# TODO: num landmark / landmark rejection count stats for debugging, should also be well integrated with VIORerunLogger for easy debugging
class VIO:
    def __init__(
        self,
        config: VIOConfig,
        feature_tracker: KLTFeatureTracker | KLTFeatureTrackerCpp,
        relative_pose_initializer: RelativePnPInitializer,
        ba: FixedLagBundleAdjuster,
        imu_preintegrator: ImuPreintegrator,
        logger: "VIORerunLogger | None" = None,
    ):
        self.config = config
        self.feature_tracker = feature_tracker
        self.T_current_from_latest_keyframe_initializer = relative_pose_initializer
        self.ba = ba
        self.imu_preintegrator = imu_preintegrator
        self.logger = logger
        self.keyframe_interval = config.keyframe_interval

        self.calibration = compute_vio_calibration(config)
        self.sgbm = SGBM() # TODO: configure sgbm via hydra too. definitely need to support max depth at least

        self.frame_idx = 0
        self.T_current_from_latest_keyframe = gtsam.Pose3.Identity() # relative pose from the latest keyframe to the current frame

        self.latest_keyframe_landmarks = []
        self.latest_all_landmarks = []

    @profile
    def process(
        self,
        timestamp: float,
        left_rect: np.ndarray,
        right_rect: np.ndarray,
        imu_acc: np.ndarray,
        imu_gyro: np.ndarray,
        imu_ts: np.ndarray | None = None,
        imu_dts: np.ndarray | None = None,
    ) -> VIOEstimate:
        """
        Process a single frame with IMU measurements.
        
        Args:
            timestamp: Frame timestamp
            left_rect: Rectified left image
            right_rect: Rectified right image
            imu_acc: IMU linear accelerations (N x 3)
            imu_gyro: IMU angular velocities (N x 3)
            imu_ts: IMU timestamps (N,)
        
        Returns:
            VIOEstimate with pose, velocity, and timestamp
        """

        # increment frame index
        self.frame_idx += 1
        is_keyframe = self.frame_idx % self.keyframe_interval == 0

        # Calculate time deltas from timestamps
        if imu_dts is not None:
            pass
        elif imu_ts is not None:
            imu_dts = np.diff(imu_ts, prepend=self.prev_imu_ts)
            self.prev_imu_ts = imu_ts[-1]  # Update to last timestamp
        else:
            raise ValueError("Must provide either imu_ts or imu_dts")
        
        # Preintegrate IMU measurements
        self.imu_preintegrator.integrate_batch(
            linear_accelerations=imu_acc,
            angular_velocities=imu_gyro,
            dts=imu_dts,
        )
        
        rect_frame = self.preprocess_frame(left_rect, right_rect)

        observations = self.feature_tracker.track_frame(rect_frame)        
        pnp_result = self.T_current_from_latest_keyframe_initializer.process_frame(
            frame_index=self.frame_idx,
            rectified_frame=rect_frame,
            track_observations=observations
        )
        assert pnp_result is not None, "PnP result should not be None, make sure you called reset() first"
        if pnp_result["status"] != "success":
            # apply constant velocity from the last frame to estimate the relative pose
            # TODO: factor out some of these repeated velocity calculations into composoable functions
            last_velocity = self.ba.get_latest_velocity()
            dt = timestamp - self.ba.prev_ts
            assert dt > 0, f"dt must be positive, got {dt}"
            
            T_current_from_previous_current = gtsam.Pose3(gtsam.Rot3.Identity(), last_velocity * dt)
        else:
            T_current_from_previous_current = pnp_result["relative_pose"]

        # compose the relative pose with the current keyframe relative pose
        self.T_current_from_latest_keyframe = self.T_current_from_latest_keyframe * T_current_from_previous_current

        # only process keyframes
        if not is_keyframe:
            pose = self.ba.get_latest_pose() * self.T_current_from_latest_keyframe
            trajectory = self.ba.get_trajectory() + [pose] # add the latest pose to the keyframed trajectory

            if self.logger is not None and self.frame_idx % self.config.log_every == 0:
                self.logger.log_step(
                    frame_idx=self.frame_idx,
                    timestamp=timestamp,
                    pose=pose,
                    frame=rect_frame,
                    trajectory=trajectory,
                    observations=observations,
                    landmarks=self.latest_keyframe_landmarks,
                    all_landmarks=self.latest_all_landmarks,
                    ba_stats=None,
                )

            T = pose.translation()
            R = pose.rotation().matrix()
            v = self.ba.get_latest_velocity()

            return VIOEstimate(
                timestamp=timestamp,
                t=T,
                R=R,
                v=v,
                keyframe=False,
            )

        # only use pnp inliers
        if self.config.ransac_inliers_only and pnp_result["status"] == "success":
            valid_track_ids = pnp_result["track_ids"]
            observations = { k: v for k, v in observations.items() if k in valid_track_ids }

        dt = timestamp - self.ba.prev_ts
        assert dt > 0, f"dt must be positive, got {dt}"

        estimated_velocity = finite_difference_velocity(
            prev_pose=gtsam.Pose3.Identity(),
            next_pose=T_current_from_previous_current,
            dt=dt
        )

        ba_stats = self.ba.process(
            frame=rect_frame,
            ts=timestamp,
            relative_pose=self.T_current_from_latest_keyframe,
            estimated_velocity=estimated_velocity,
            landmark_observations=observations,
            pim=self.imu_preintegrator.pim,
            optimize=(self.ba.frame_idx % self.config.optimize_every == 0),
            profile=False,
        )

        # reset relative pose since this is a new keyframe
        assert is_keyframe
        self.T_current_from_latest_keyframe = gtsam.Pose3.Identity()
        
        # Update IMU preintegrator with latest bias estimate
        self.imu_preintegrator.reset(self.ba.get_bias())
        
        latest_pose = self.ba.get_latest_pose()
        latest_velocity = self.ba.get_latest_velocity()
        trajectory = self.ba.get_trajectory()

        if self.logger is not None and self.frame_idx % self.config.log_landmarks_every == 0:
            self.latest_keyframe_landmarks = self.ba.get_active_landmarks()
            self.latest_all_landmarks = self.ba.get_all_landmarks()

        if self.logger is not None and self.frame_idx % self.config.log_every == 0:
            self.logger.log_step(
                frame_idx=self.frame_idx,
                timestamp=timestamp,
                pose=latest_pose,
                frame=rect_frame,
                trajectory=trajectory,
                observations=observations,
                landmarks=self.latest_keyframe_landmarks,
                all_landmarks=self.latest_all_landmarks,
                ba_stats=ba_stats,
                bias=self.ba.get_bias(),
            )

        t = latest_pose.translation()
        R = latest_pose.rotation().matrix()
        v = latest_velocity
        
        return VIOEstimate(timestamp=timestamp, t=t, R=R, v=v, keyframe=True)

    
    def reset(self, timestamp: float, left_rect: np.ndarray, right_rect: np.ndarray, t: np.ndarray, R: np.ndarray, v: np.ndarray = None):
        self.feature_tracker.reset()
        initial_pose = gtsam.Pose3(gtsam.Rot3(R), t)
        self.T_current_from_latest_keyframe_initializer.reset(initial_pose)

        rect_frame = self.preprocess_frame(left_rect, right_rect)
        observations = self.feature_tracker.track_frame(rect_frame)
        pnp_result = self.T_current_from_latest_keyframe_initializer.process_frame(
            frame_index=0,
            rectified_frame=rect_frame,
            track_observations=observations
        )
        assert pnp_result is None

        if v is None:
            v = np.zeros(3)
        self.ba.reset(ts=timestamp, pose=gtsam.Pose3(gtsam.Rot3(R), t), velocity=v)
        
        # Initialize IMU timestamp tracking
        self.prev_imu_ts = timestamp

        estimate = VIOEstimate(timestamp=timestamp, t=t, R=R, v=v, keyframe=True)
        return estimate


    def get_estimated_trajectory(self) -> list[VIOEstimate]:
        estimates = []
        
        poses = self.ba.get_trajectory()
        velocities = self.ba.get_trajectory_velocities()
        
        for i in range(len(poses)):
            t = poses[i].translation()
            R = poses[i].rotation().matrix()
            v = velocities[i]
            estimates.append(VIOEstimate(timestamp=self.ba.ts[i], t=t, R=R, v=v, keyframe=True))
        
        return estimates

    def get_estimated_bias(self) -> gtsam.imuBias.ConstantBias:
        return self.ba.get_bias()

    def get_distance_traveled(self) -> float:
        trajectory = self.ba.get_trajectory()
        
        total_distance = 0.0
        for i in range(len(trajectory) - 1):
            total_distance += np.linalg.norm(trajectory[i].translation() - trajectory[i + 1].translation())
        
        return total_distance
    
    def preprocess_frame(self, left_rect: np.ndarray, right_rect: np.ndarray) -> RectifiedStereoFrame:
        rect_frame = RectifiedStereoFrame(
            left=None,
            right=None,
            left_rect=left_rect,
            right_rect=right_rect,
            calibration=self.calibration,
        )

        return rect_frame

    @staticmethod
    def from_config(
        config_path: str | Path,
        overrides: Sequence[str] | None = None,
        **override_kwargs: object,
    ) -> "VIO":
        cfg = compose_config(config_path, overrides=overrides, **override_kwargs)
        vio_cfg = extract_target_config(cfg, context=str(config_path))
        return instantiate(vio_cfg)
