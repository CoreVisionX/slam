from dataclasses import dataclass
import numpy as np
import gtsam
from slam.registration.registration import StereoCalibration

@dataclass
class VIOEstimate:
    timestamp: float
    t: np.ndarray
    R: np.ndarray
    v: np.ndarray
    keyframe: bool

@dataclass
class VIOCalibration(StereoCalibration):
    imu_from_left: gtsam.Pose3
    imu_from_right: gtsam.Pose3
