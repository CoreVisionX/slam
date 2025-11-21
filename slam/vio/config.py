from dataclasses import dataclass
import numpy as np
import gtsam

def _rectified_q_matrix(
    K_left_rect: np.ndarray,
    K_right_rect: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """
    Construct the reprojection matrix for a rectified stereo pair using standard
    OpenCV conventions (see reprojectImageTo3D documentation).
    """
    Tx = float(np.asarray(T, dtype=np.float64).ravel()[0])
    if np.isclose(Tx, 0.0):
        raise ValueError("Baseline is zero; cannot build reprojection matrix.")

    cx_left = float(K_left_rect[0, 2])
    cy_left = float(K_left_rect[1, 2])
    cx_right = float(K_right_rect[0, 2])
    fx = float(K_left_rect[0, 0])

    Q = np.array([
        [1.0, 0.0, 0.0, -cx_left],
        [0.0, 1.0, 0.0, -cy_left],
        [0.0, 0.0, 0.0, fx],
        [0.0, 0.0, -1.0 / Tx, (cx_left - cx_right) / Tx],
    ], dtype=np.float64)
    return Q

from .types import VIOCalibration

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

    # logging
    log_every: int

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


def compute_vio_calibration(config: VIOConfig) -> VIOCalibration:
    T = np.array([config.baseline, 0.0, 0.0])
    Q = _rectified_q_matrix(
        K_left_rect=config.K_left_rect,
        K_right_rect=config.K_right_rect,
        T=T,
    )

    calib = VIOCalibration(
        K_left_rect=config.K_left_rect,
        K_right_rect=config.K_right_rect,
        Q=Q,
        T=T,
        R=np.eye(3),
        width=config.width,
        height=config.height,
        imu_from_left=gtsam.Pose3(config.imu_from_left),
        imu_from_right=gtsam.Pose3(config.imu_from_right),

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
