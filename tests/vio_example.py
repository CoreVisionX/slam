from slam.local_vo.bundle_adjustment import finite_difference_velocity
from tests.test_utils import se3_inverse
import gtsam
from tests.test_utils import EuRoCStereoCalibration
from slam.local_vo.bundle_adjustment import FixedLagBundleAdjuster, BundleAdjustmentConfig
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
from typing import Mapping, Sequence

import rerun as rr

from slam.local_vo.klt_tracker import KLTFeatureTracker, TrackObservation
from slam.local_vo.relative_pose import RelativePnPInitializer
from tests.test_utils import _rectified_q_matrix, estimate_world_gravity_from_first_batch
from viz import rr_log_pose, rr_log_trajectory
from registration.registration import RectifiedStereoFrame
from registration.registration import StereoCalibration
from registration.registration import StereoDepthFrame
from depth.sgbm import SGBM
from tests.datasets.pipeline import SequencePreprocessor

from dataclasses import dataclass
import numpy as np

# VIO pipeline (move this into their own files after you finish a basic implementation)
# TODO: move all TODOs and improvements into more atomic tasks you can assign to AI or chunk out into a time box to knock out as incremental improvements
# TODO: see if CombinedImMeasurements helps so bias can drift

@dataclass
class VIOConfig:
    # imu
    gravity: np.ndarray
    imu_from_left: np.ndarray
    imu_from_right: np.ndarray

    # stereo
    baseline: float
    
    # pinhole calibration
    K_left_rect: np.ndarray
    K_right_rect: np.ndarray

    # image dimensions
    width: int
    height: int

    # bundle adjustment
    optimize_every: int

    def __post_init__(self):
        if not isinstance(self.gravity, np.ndarray):
            self.gravity = np.array(self.gravity)
        if not isinstance(self.imu_from_left, np.ndarray):
            self.imu_from_left = np.array(self.imu_from_left)
        if not isinstance(self.imu_from_right, np.ndarray):
            self.imu_from_right = np.array(self.imu_from_right)
        if not isinstance(self.K_left_rect, np.ndarray):
            self.K_left_rect = np.array(self.K_left_rect)
        if not isinstance(self.K_right_rect, np.ndarray):
            self.K_right_rect = np.array(self.K_right_rect)


# remember, composition over inheritance. don't put something in a class and repeat something you don't need to
# just write modular code you can reuse. don't rewrite the same tedious code a bunch of times. Don't be scared to refactor!
# that's how tech debt gets you
def compute_vio_calibration(config: VIOConfig) -> StereoCalibration:
    T = np.array([config.baseline, 0.0, 0.0])
    Q = _rectified_q_matrix(
        K_left_rect=config.K_left_rect,
        K_right_rect=config.K_right_rect,
        T=T,
    )

    calib = EuRoCStereoCalibration(
        K_left_rect=config.K_left_rect,
        K_right_rect=config.K_right_rect,
        Q=Q,
        T=T,
        R=np.eye(3),
        width=config.width,
        height=config.height,
        T_B_from_S0=gtsam.Pose3(config.imu_from_left),
        T_B_from_S1=gtsam.Pose3(config.imu_from_right),

        # TODO: create a simple stereo calibration class and get rid of all this junk
        # everything else will just set to none right now, they shouldn't be needed for already rectified frames
        K_left=None,
        K_right=None,
        D_left=None,
        D_right=None,
        R_left=None,
        R_right=None,
        P_left=None,
        P_right=None,
        map_left_x=None,
        map_left_y=None,
        map_right_x=None,
        map_right_y=None,
    )

    return calib


@dataclass
class VIOEstimate:
    timestamp: float
    t: np.ndarray
    R: np.ndarray
    v: np.ndarray


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
        logger: "VIORerunLogger | None" = None,
    ):
        self.config = config
        self.feature_tracker = feature_tracker
        self.relative_pose_initializer = relative_pose_initializer
        self.ba = ba
        self.logger = logger

        self.calibration = compute_vio_calibration(config)
        self.sgbm = SGBM() # TODO: configure sgbm via hydra too. definitely need to support max depth at least

    # TODO: do imu preintegration in here
    def process(self, timestamp: float, left_rect: np.ndarray, right_rect: np.ndarray, pim: gtsam.PreintegratedImuMeasurements):
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

        self.ba.process(
            frame=depth_frame,
            ts=timestamp,
            relative_pose=relative_pose,
            estimated_velocity=estimated_velocity,
            landmark_observations=observations,
            pim=pim,
            optimize=(self.ba.frame_idx % self.config.optimize_every == 0),
            profile=False,
        )
        latest_pose = self.ba.get_latest_pose()
        latest_velocity = self.ba.get_latest_velocity()
        trajectory = self.ba.get_trajectory()

        if self.logger is not None:
            self.logger.log_step(
                frame_idx=self.ba.frame_idx,
                timestamp=timestamp,
                pose=latest_pose,
                frame=depth_frame,
                trajectory=trajectory,
                observations=observations,
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

# TODO: logging information like how many observations were added, how many were rejected, how many active landmarks there are, 3D logs of the landmarks as the evolve throughout the sequence will all be super helpful for debugging
# TODO: make sure it logs the KLT features in 2D space like klt.py with correct class ids assigned. It would be so awesome if you could assign the same class ids to the 3D landmarks as well
# this is a prime example of something you should be careful to do composition over inheritance for
# don't rewrite a bunch of rerun logging functions because you do low level rerun logging the same way in a bunch
# of different domain specific classes
class VIORerunLogger:
    def __init__(
        self,
        *,
        app_id: str = "vio_example",
        base_path: str = "vio",
        spawn: bool = True,
        view_coordinates: rr.ViewCoordinates = rr.ViewCoordinates.RIGHT_HAND_X_UP,
        trajectory_thickness: float = 0.008,
        trajectory_color: tuple[int, int, int] = (0, 0, 255),
    ) -> None:
        rr.init(app_id, spawn=spawn)

        self._base_path = base_path.rstrip("/")
        if not self._base_path:
            self._base_path = "vio"

        self.view_coordinates = view_coordinates
        self.trajectory_thickness = trajectory_thickness
        self.trajectory_color = trajectory_color

    def log_step(
        self,
        *,
        frame_idx: int,
        timestamp: float,
        pose: gtsam.Pose3,
        frame: RectifiedStereoFrame | StereoDepthFrame,
        trajectory: Sequence[gtsam.Pose3],
        observations: Mapping[int, TrackObservation],
    ) -> None:
        """Log the current VIO step to rerun."""
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("timestamp", timestamp=timestamp)
        self._log_pose(pose, frame)
        self._log_trajectory(trajectory)
        self._log_klt_features(observations)

    def _log_pose(self, pose: gtsam.Pose3, frame: RectifiedStereoFrame | StereoDepthFrame) -> None:
        rr_log_pose(f"{self._base_path}/pose", pose, frame, camera_xyz=self.view_coordinates)

    def _log_trajectory(self, trajectory: Sequence[gtsam.Pose3]) -> None:
        if not trajectory:
            return
        rr_log_trajectory(f"{self._base_path}/trajectory", list(trajectory), radii=self.trajectory_thickness, color=self.trajectory_color)

    def _log_klt_features(self, observations: Mapping[int, TrackObservation]) -> None:
        image_path = f"{self._base_path}/pose/rgb"
        observation_count = len(observations)

        if observation_count == 0:
            rr.log(image_path, rr.Points2D(np.empty((0, 2), dtype=np.float32)))
            rr.log(f"{self._base_path}/klt/n_observations", rr.Scalars(0))
            return

        obs_items = list(observations.items())
        points = np.array([obs.keypoint for _, obs in obs_items], dtype=np.float32)
        class_ids = [track_id for track_id, _ in obs_items]
        rr.log(
            image_path,
            rr.Points2D(points, radii=3.0, class_ids=class_ids),
        )
        rr.log(
            f"{self._base_path}/klt/n_observations",
            rr.Scalars(observation_count),
        )


# TODO: factor out into an output/eval utils file or something
def save_tum_sequence(vio_outputs: list[VIOEstimate], output_path: str):
    """Save VIO trajectory in TUM format (timestamp tx ty tz qx qy qz qw)."""
    if not vio_outputs:
        print("No VIO outputs to save")
        return
    
    rows = []
    for output in vio_outputs:
        # Convert rotation matrix to quaternion
        rot = gtsam.Rot3(output.R)
        quat = rot.toQuaternion()
        
        rows.append([
            output.timestamp,
            float(output.t[0]),
            float(output.t[1]),
            float(output.t[2]),
            float(quat.x()),
            float(quat.y()),
            float(quat.z()),
            float(quat.w()),
        ])
    
    tum_data = np.asarray(rows, dtype=np.float64)
    np.savetxt(output_path, tum_data, fmt="%.9f")
    print(f"Saved TUM trajectory: {output_path}")

# TODO: factor out imu preintegrator class
# TODO: none of the hydra instansiation should be user facing, that should be hidden behind some easy setup methods.
# something like the TUM output saving shouldn't be a part of the config. it should just be an argparse option
# TODO: examples of per sequence/environment configs might be good

@hydra.main(version_base=None, config_path="config", config_name="vio_config")
def main(cfg: DictConfig):
    # Instantiate VIO directly from config
    vio: VIO = instantiate(cfg.vio)
    
    # Load data pipeline
    pipeline: SequencePreprocessor = instantiate(cfg.data_pipeline, _convert_="partial")
    preprocessed = pipeline.prepare(seed=13, max_depth=40.0)
    rectified_frames = preprocessed.rectified_frames
    sequence = preprocessed.sequence
    
    # Setup IMU Preintegration Params
    # TODO: move the imu related params and logic into their own dedicated IMU preintegrator class
    ba_config = vio.ba.config

    # gravity
    imu_params = gtsam.PreintegrationParams.MakeSharedU(np.linalg.norm(vio.config.gravity))
    imu_params.n_gravity = vio.config.gravity

    # noise
    imu_params.setAccelerometerCovariance(np.eye(3) * (ba_config.imu_accel_noise**2))
    imu_params.setGyroscopeCovariance(np.eye(3) * (ba_config.imu_gyro_noise**2))
    imu_params.setIntegrationCovariance(np.eye(3) * (ba_config.imu_integration_noise**2))

    bias = gtsam.imuBias.ConstantBias()

    pim = gtsam.PreintegratedImuMeasurements(imu_params, bias)
    
    print(f"Processing {len(rectified_frames)} frames...")

   # Reset VIO
    gt_pose = sequence.world_poses[0]
    
    # Estimate initial velocity
    next_gt_pose = sequence.world_poses[1]
    dt = sequence.frame_timestamps[1] - sequence.frame_timestamps[0]
    first_velocity = finite_difference_velocity(gt_pose, next_gt_pose, dt)
    
    vio.reset(
        timestamp=sequence.frame_timestamps[0],
        left_rect=rectified_frames[0].left_rect,
        right_rect=rectified_frames[0].right_rect,
        t=gt_pose.translation(),
        R=gt_pose.rotation().matrix(),
        v=first_velocity
    )
    
    # Process each frame in the sequence
    for idx in tqdm(range(1, len(rectified_frames)), desc="Estimating trajectory"):
        rect_frame = rectified_frames[idx]
        timestamp = sequence.frame_timestamps[idx]
        
        # Preintegrate IMU measurements
        batch = sequence.imu_measurements[idx]
        for i in range(len(batch)):
            pim.integrateMeasurement(
                batch.linear_accelerations[i],
                batch.angular_velocities[i],
                deltaT=batch.dts[i],
            )

        # Process frame
        output = vio.process(
            timestamp=timestamp,
            left_rect=rect_frame.left_rect,
            right_rect=rect_frame.right_rect,
            pim=pim
        )
        
        pim.resetIntegrationAndSetBias(vio.get_estimated_bias())
    
    # Save TUM trajectory
    output_path = Path(__file__).parent / "results" / "vio_estimated.txt"
    save_tum_sequence(vio.get_estimated_trajectory(), str(output_path))

if __name__ == "__main__":
    main()
