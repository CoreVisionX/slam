

from dataclasses import dataclass
from typing import Generic, TypeVar
import gtsam
import numpy as np

from dataclasses import dataclass
import numpy as np
import cv2
import kornia.feature as KF

@dataclass
class StereoCalibration:
    K_left: np.ndarray
    K_right: np.ndarray
    K_left_rect: np.ndarray
    K_right_rect: np.ndarray
    D_left: np.ndarray
    D_right: np.ndarray
    R: np.ndarray           # right w.r.t. left
    T: np.ndarray           # (3,1) right w.r.t. left, meters
    R_left: np.ndarray
    R_right: np.ndarray
    P_left: np.ndarray
    P_right: np.ndarray
    Q: np.ndarray
    map_left_x: np.ndarray
    map_left_y: np.ndarray
    map_right_x: np.ndarray
    map_right_y: np.ndarray
    width: int
    height: int

    def resize(self, new_width: int, new_height: int) -> 'StereoCalibration':
        s_x = new_width / self.width
        s_y = new_height / self.height
        K_left_rect = np.array([
            [self.K_left_rect[0, 0] * s_x, 0, self.K_left_rect[0, 2] * s_x],
            [0, self.K_left_rect[1, 1] * s_y, self.K_left_rect[1, 2] * s_y],
            [0, 0, 1]
        ])
        K_right_rect = np.array([
            [self.K_right_rect[0, 0] * s_x, 0, self.K_right_rect[0, 2] * s_x],
            [0, self.K_right_rect[1, 1] * s_y, self.K_right_rect[1, 2] * s_y],
            [0, 0, 1]
        ])

        return StereoCalibration(
            K_left=self.K_left, K_right=self.K_right,
            K_left_rect=K_left_rect, K_right_rect=K_right_rect,
            D_left=self.D_left, D_right=self.D_right,
            R=self.R, T=self.T,
            R_left=self.R_left, R_right=self.R_right,
            P_left=self.P_left, P_right=self.P_right,
            Q=self.Q,
            map_left_x=self.map_left_x, map_left_y=self.map_left_y,
            map_right_x=self.map_right_x, map_right_y=self.map_right_y,
            width=new_width, height=new_height
        )

    @classmethod
    def create(cls, K, T, R, width: int, height: int) -> 'StereoCalibration':
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError("K must be 3x3.")

        # zero distortion unless you have real coeffs
        D = np.zeros((5, 1), dtype=np.float64)

        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError("R must be 3x3.")

        T = np.asarray(T, dtype=np.float64).reshape(3, 1)
        if T.shape != (3, 1):
            raise ValueError("T must be length-3 or shape (3,1).")

        # infer image size from principal point (assumes centered intrinsics)
        cx, cy = float(K[0, 2]), float(K[1, 2])
        width, height = int(round(2 * cx)), int(round(2 * cy))
        if width <= 0 or height <= 0:
            raise ValueError("Could not infer image size from K; pass a centered K.")

        image_size = (width, height)

        # Rectification (OpenCV convention; zero disparity so principal points align)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K, D, K, D, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )

        # Rectification maps
        map_lx, map_ly = cv2.initUndistortRectifyMap(
            K, D, R1, P1, image_size, cv2.CV_32FC1
        )
        map_rx, map_ry = cv2.initUndistortRectifyMap(
            K, D, R2, P2, image_size, cv2.CV_32FC1
        )

        # Rectified intrinsics
        K_left_rect = P1[:3, :3]
        K_right_rect = P2[:3, :3]

        return cls(
            K_left=K.copy(), K_right=K.copy(),
            D_left=D.copy(), D_right=D.copy(),
            K_left_rect=K_left_rect, K_right_rect=K_right_rect,
            R=R.copy(), T=T.copy(),
            R_left=R1, R_right=R2,
            P_left=P1, P_right=P2,
            Q=Q,
            map_left_x=map_lx, map_left_y=map_ly,
            map_right_x=map_rx, map_right_y=map_ry,
            width=width, height=height
        )

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if left.shape[:2] != right.shape[:2]:
            raise ValueError("Left/right images must have the same shape.")

        h, w = left.shape[:2]
        # Recompute maps on-the-fly if image size differs from stored maps
        if self.map_left_x.shape[:2] != (h, w):
            size = (w, h)
            self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
                self.K_left, self.D_left, self.R_left, self.P_left, size, cv2.CV_32FC1
            )
            self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
                self.K_right, self.D_right, self.R_right, self.P_right, size, cv2.CV_32FC1
            )

        left_r = cv2.remap(left,  self.map_left_x,  self.map_left_y,
                           interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        right_r = cv2.remap(right, self.map_right_x, self.map_right_y,
                            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return left_r, right_r

@dataclass
class StereoFrame:
    left: np.ndarray
    right: np.ndarray
    calibration: StereoCalibration

    def rectify(self) -> 'RectifiedStereoFrame':
        left_rect, right_rect = self.calibration.rectify(self.left, self.right)
    
        return RectifiedStereoFrame(
            left=self.left,
            right=self.right,
            left_rect=left_rect,
            right_rect=right_rect,
            calibration=self.calibration
        )

@dataclass
class RectifiedStereoFrame(StereoFrame):
    left_rect: np.ndarray
    right_rect: np.ndarray

    def resize(self, new_width: int, new_height: int) -> 'RectifiedStereoFrame':
        left_rect = cv2.resize(self.left_rect, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        right_rect = cv2.resize(self.right_rect, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        calibration = self.calibration.resize(new_width, new_height)

        return RectifiedStereoFrame(
            left=self.left,
            right=self.right,
            left_rect=left_rect,
            right_rect=right_rect,
            calibration=calibration,
        )

@dataclass
class StereoDepthFrame(RectifiedStereoFrame):
    left_depth: np.ndarray
    left_depth_xyz: np.ndarray


F = TypeVar("F")

# tbh composition over inheritance would make this easier
@dataclass
class FeatureFrame(StereoDepthFrame, Generic[F]):
    # discard the memory intensive images to save memory
    left: None
    right: None
    left_rect: None
    right_rect: None
    left_depth: None
    left_depth_xyz: None
    features: F


S = TypeVar("S")

@dataclass
class FramePair(Generic[S]):
    first: S
    second: S

@dataclass
class MatchedFramePair(FramePair[S], Generic[S]):
    first: S
    second: S
    matches: np.ndarray

    @property
    def mkpts1(self) -> np.ndarray:
        return self.first.features['keypoints'][self.matches[:, 0], :]

    @property
    def mkpts2(self) -> np.ndarray:
        return self.second.features['keypoints'][self.matches[:, 1], :]

    @property
    def mkpts1_3d(self) -> np.ndarray:
        return self.first.features['keypoints_3d'][self.matches[:, 0]]

    @property
    def mkpts1_depth(self) -> np.ndarray:
        return self.first.features['keypoints_depth'][self.matches[:, 0]]

    @property
    def mkpts1_color(self) -> np.ndarray:
        return self.first.features['keypoints_color'][self.matches[:, 0]]

@dataclass
class IndexedFramePair(FramePair[S], Generic[S]):
    first_idx: int
    second_idx: int

@dataclass
class FramePairWithGroundTruth(FramePair[S], Generic[S]):
    first_T_second: "gtsam.Pose3"
