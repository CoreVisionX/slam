from tests.test_utils import se3_inverse
import gtsam
from slam.local_vo.bundle_adjustment import finite_difference_velocity
from tests.test_utils import EuRoCStereoCalibration
from slam.local_vo.bundle_adjustment import FixedLagBundleAdjuster
from slam.local_vo.klt_tracker import KLTFeatureTracker
from slam.local_vo.relative_pose import RelativePnPInitializer
from tests.test_utils import _rectified_q_matrix
from registration.registration import RectifiedStereoFrame
from registration.registration import StereoCalibration
from registration.registration import StereoFrame
from registration.registration import StereoDepthFrame
from depth.sgbm import SGBM
from dataclasses import dataclass
import numpy as np

# VIO pipeline (move this into their own files after you finish a basic implementation)
# TODO: move all TODOs and improvements into more atomic tasks you can assign to AI or chunk out into a time box to knock out as incremental improvements

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
class VIOOutput:
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
    def __init__(self, config: VIOConfig, feature_tracker: KLTFeatureTracker, relative_pose_initializer: RelativePnPInitializer, ba: FixedLagBundleAdjuster):
        self.config = config
        self.feature_tracker = feature_tracker
        self.relative_pose_initializer = relative_pose_initializer
        self.ba = ba

        self.calibration = compute_vio_calibration(config)
        self.sgbm = SGBM() # TODO: configure sgbm via hydra too
        self.frame_count = 0

    # TODO: do imu preintegration in here
    def process(self, timestamp: float, left_rect: np.ndarray, right_rect: np.ndarray, pim: gtsam.PreintegratedImuMeasurements):
        rect_frame, depth_frame = self.preprocess_frame(left_rect, right_rect)

        observations = self.feature_tracker.track_frame(rect_frame, depth_frame)        
        pnp_result = self.relative_pose_initializer.process_frame(
            frame_index=self.frame_count,
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

        estimated_velocity = finite_difference_velocity(gtsam.Pose3.Identity(), relative_pose, dt)

        self.ba.process(
            frame=depth_frame,
            ts=timestamp,
            relative_pose=relative_pose,
            estimated_velocity=estimated_velocity,
            landmark_observations=observations,
            pim=pim,
            optimize=(self.frame_count % self.config.optimize_every == 0),
            profile=False,
        )
        latest_pose = self.ba.get_latest_pose()
        latest_velocity = self.ba.get_latest_velocity()

        t = latest_pose.translation()
        R = latest_pose.rotation().matrix()
        v = latest_velocity
        
        self.frame_count += 1

        return VIOOutput(timestamp=timestamp, t=t, R=R, v=v)

    
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

        self.frame_count = 0
        
    
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
    pass

# <100 line VIO example

# maybe factor this part out into the library?
def load_euroc_vio_config(calib_path: str) -> VIOConfig:
    pass

# TODO: maybe all of this should be loaded into VIOConfig via hydra tbh
imu_from_left = np.array([0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
imu_from_right = np.array([0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
         0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
        -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
         0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
right_from_left = se3_inverse(imu_from_right) @ imu_from_left
baseline = np.linalg.norm(right_from_left[:3, 3]) # TODO: see if taking the translation along the baseline axis is better

euroc_config = VIOConfig(
    gravity=np.array([0.0, 9.81, 0.0]),
    imu_from_left=imu_from_left,
    imu_from_right=imu_from_right,
    baseline=baseline,
    K_left_rect=np.array([
        [458.654, 0.0, 367.215],
        [0.0, 457.296, 248.375],
        [0.0, 0.0, 1.0]
    ]),
    K_right_rect=np.array([
        [457.587, 0.0, 379.999],
        [0.0, 456.134, 255.238],
        [0.0, 0.0, 1.0]
    ]),
    width=752,
    height=480,
    optimize_every=4,
)
    
