from pathlib import Path
from typing import Sequence

import gtsam
import numpy as np
from hydra.utils import instantiate
from slam.vio.bundle_adjustment import FixedLagBundleAdjuster, finite_difference_velocity
from slam.vio.klt_tracker import KLTFeatureTracker
from slam.vio.relative_pose import RelativePnPInitializer
from registration.registration import RectifiedStereoFrame, StereoDepthFrame
from depth.sgbm import SGBM
from .config import VIOConfig, compute_vio_calibration
from .types import VIOEstimate
from .io import VIORerunLogger
from .imu_preintegration import ImuPreintegrator
from slam.hydra_utils import compose_config, extract_target_config

# TODO: see if switching to CombinedImuFactors helps on longer sequences by accounting for IMU bias drift
# TODO: see if relative pose initialization via Imu Preintegration is better than PnP

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
        feature_tracker: KLTFeatureTracker,
        relative_pose_initializer: RelativePnPInitializer,
        ba: FixedLagBundleAdjuster,
        imu_preintegrator: ImuPreintegrator,
        logger: "VIORerunLogger | None" = None,
    ):
        self.config = config
        self.feature_tracker = feature_tracker
        self.relative_pose_initializer = relative_pose_initializer
        self.ba = ba
        self.imu_preintegrator = imu_preintegrator
        self.logger = logger

        self.calibration = compute_vio_calibration(config)
        self.sgbm = SGBM() # TODO: configure sgbm via hydra too. definitely need to support max depth at least

    def process(
        self,
        timestamp: float,
        left_rect: np.ndarray,
        right_rect: np.ndarray,
        imu_acc: np.ndarray,
        imu_gyro: np.ndarray,
        imu_ts: np.ndarray,
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
        # Calculate time deltas from timestamps
        imu_dts = np.diff(imu_ts, prepend=self.prev_imu_ts)
        self.prev_imu_ts = imu_ts[-1]  # Update to last timestamp
        
        # Preintegrate IMU measurements
        self.imu_preintegrator.integrate_batch(
            linear_accelerations=imu_acc,
            angular_velocities=imu_gyro,
            dts=imu_dts,
        )
        
        rect_frame, depth_frame = self.preprocess_frame(left_rect, right_rect)

        observations = self.feature_tracker.track_frame(rect_frame, depth_frame)        
        pnp_result = self.relative_pose_initializer.process_frame(
            frame_index=self.ba.frame_idx + 1,
            rectified_frame=rect_frame,
            depth_frame=depth_frame,
            track_observations=observations
        )
        assert pnp_result is not None, "PnP result should not be None, make sure you called reset() first"
        if pnp_result["status"] != "success":
            # apply constant velocity from the last frame to estimate the relative pose
            # TODO: factor out some of these repeated velocity calculations into composoable functions
            last_velocity = self.ba.get_latest_velocity()
            dt = timestamp - self.ba.prev_ts
            assert dt > 0, f"dt must be positive, got {dt}"
            
            relative_pose = gtsam.Pose3(gtsam.Rot3.Identity(), last_velocity * dt)
        else:
            relative_pose = pnp_result["relative_pose"]

        dt = timestamp - self.ba.prev_ts
        assert dt > 0, f"dt must be positive, got {dt}"

        estimated_velocity = finite_difference_velocity(
            prev_pose=gtsam.Pose3.Identity(),
            next_pose=relative_pose,
            dt=dt
        )

        ba_stats = self.ba.process(
            frame=depth_frame,
            ts=timestamp,
            relative_pose=relative_pose,
            estimated_velocity=estimated_velocity,
            landmark_observations=observations,
            pim=self.imu_preintegrator.pim,
            optimize=(self.ba.frame_idx % self.config.optimize_every == 0),
            profile=False,
        )
        
        # Update IMU preintegrator with latest bias estimate
        self.imu_preintegrator.reset(self.ba.get_bias())
        
        latest_pose = self.ba.get_latest_pose()
        latest_velocity = self.ba.get_latest_velocity()
        trajectory = self.ba.get_trajectory()

        if self.logger is not None and self.ba.frame_idx % self.config.log_every == 0:
            self.logger.log_step(
                frame_idx=self.ba.frame_idx,
                timestamp=timestamp,
                pose=latest_pose,
                frame=depth_frame,
                trajectory=trajectory,
                observations=observations,
                landmarks=self.ba.get_active_landmarks(),
                ba_stats=ba_stats,
            )

        t = latest_pose.translation()
        R = latest_pose.rotation().matrix()
        v = latest_velocity
        
        return VIOEstimate(timestamp=timestamp, t=t, R=R, v=v)

    
    def reset(self, timestamp: float, left_rect: np.ndarray, right_rect: np.ndarray, t: np.ndarray, R: np.ndarray, v: np.ndarray = None):
        self.feature_tracker.reset()
        initial_pose = gtsam.Pose3(gtsam.Rot3(R), t)
        self.relative_pose_initializer.reset(initial_pose)

        rect_frame, depth_frame = self.preprocess_frame(left_rect, right_rect)
        observations = self.feature_tracker.track_frame(rect_frame, depth_frame)
        pnp_result = self.relative_pose_initializer.process_frame(
            frame_index=0,
            rectified_frame=rect_frame,
            depth_frame=depth_frame,
            track_observations=observations
        )
        assert pnp_result is None

        if v is None:
            v = np.zeros(3)
        self.ba.reset(ts=timestamp, pose=gtsam.Pose3(gtsam.Rot3(R), t), velocity=v)
        
        # Initialize IMU timestamp tracking
        self.prev_imu_ts = timestamp

        estimate = VIOEstimate(timestamp=timestamp, t=t, R=R, v=v)
        return estimate


    def get_estimated_trajectory(self) -> list[VIOEstimate]:
        estimates = []
        
        poses = self.ba.get_trajectory()
        velocities = self.ba.get_trajectory_velocities()
        
        for i in range(len(poses)):
            t = poses[i].translation()
            R = poses[i].rotation().matrix()
            v = velocities[i]
            estimates.append(VIOEstimate(timestamp=self.ba.ts[i], t=t, R=R, v=v))
        
        return estimates

    def get_estimated_bias(self) -> gtsam.imuBias.ConstantBias:
        return self.ba.get_bias()
    
    def preprocess_frame(self, left_rect: np.ndarray, right_rect: np.ndarray) -> tuple[RectifiedStereoFrame, StereoDepthFrame]:
        rect_frame = RectifiedStereoFrame(
            left=None,
            right=None,
            left_rect=left_rect,
            right_rect=right_rect,
            calibration=self.calibration,
        )
        depth_frame = self.sgbm.compute_depth(rect_frame)

        return rect_frame, depth_frame

    @staticmethod
    def from_config(
        config_path: str | Path,
        overrides: Sequence[str] | None = None,
        **override_kwargs: object,
    ) -> "VIO":
        cfg = compose_config(config_path, overrides=overrides, **override_kwargs)
        vio_cfg = extract_target_config(cfg, context=str(config_path))
        return instantiate(vio_cfg)
