import cv2
import gtsam
import numpy as np
import torch
import kornia as K

from registration.registration import FramePair, RectifiedStereoFrame, StereoDepthFrame, StereoFrame

def rectify_stereo_frame_pair(pair: FramePair[StereoFrame]) -> FramePair[RectifiedStereoFrame]:
    first_rect = pair.first.rectify()
    second_rect = pair.second.rectify()

    return FramePair[RectifiedStereoFrame](
        first=first_rect,
        second=second_rect
    )

def stack_pair_images(pair: FramePair, image_attr: str) -> np.ndarray:
    assert hasattr(pair.first, image_attr) and hasattr(pair.second, image_attr)
    
    return np.hstack((getattr(pair.first, image_attr), getattr(pair.second, image_attr)))

def draw_matches(pair: FramePair[RectifiedStereoFrame], mkpts1: np.ndarray, mkpts2: np.ndarray, inlier_mask: np.ndarray = None, inlier_color: tuple[int, int, int] = (0, 255, 0), outlier_color: tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    img1 = pair.first.left_rect
    img2 = pair.second.left_rect

    combined_img = np.hstack((img1, img2))

    offset = np.array([img1.shape[1], 0])

    for idx, (i, j) in enumerate(zip(mkpts1, mkpts2)):
        if inlier_mask is not None and inlier_mask[idx]:
            color = inlier_color
        else:
            color = outlier_color

        cv2.line(combined_img, (int(i[0]), int(i[1])), (int(j[0] + offset[0]), int(j[1] + offset[1])), color, 2)

    return combined_img

def np_to_kornia(np_array: np.ndarray) -> torch.Tensor:
    assert np_array.ndim == 3 and np_array.shape[-1] == 3

    return K.image_to_tensor(np_array, keepdim=False).float() / 255.0

def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]]
    mkpts2 = kp2[idxs[:, 1]]
    return mkpts1, mkpts2


def fundamental_fitler(mkpts1: np.ndarray, mkpts2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Fm, inliers = cv2.findFundamentalMat(
        mkpts1, mkpts2, cv2.USAC_MAGSAC, 2.0, 0.999, 100000
    )
    mask = inliers.ravel() > 0

    return mkpts1[mask], mkpts2[mask], mask


def solve_pnp(pair: FramePair[StereoDepthFrame], mkpts1: np.ndarray, mkpts2: np.ndarray) -> tuple[gtsam.Pose3, np.ndarray, np.ndarray, np.ndarray]:
    u = mkpts1[:, 1].astype(int)
    v = mkpts1[:, 0].astype(int)

    mkpts1_3d = pair.first.left_depth_xyz[u, v, :]
    mkpts1_depth = pair.first.left_depth[u, v]

    valid_mask = mkpts1_depth > 0
    mkpts1_3d = mkpts1_3d[valid_mask]
    mkpts1 = mkpts1[valid_mask]
    mkpts2 = mkpts2[valid_mask]

    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
        mkpts1_3d, 
        mkpts2, 
        pair.first.calibration.K_left_rect, 
        np.zeros((5, 1)),
        reprojectionError=2.0,
        confidence=0.999,
        iterationsCount=10_000,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    inliers = inliers.ravel()

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    if not ret:
        raise ValueError("Failed to solve PnP")

    if len(inliers) < 10:
        raise ValueError("Not enough inliers to solve PnP")

    return gtsam.Pose3(gtsam.Rot3(R), t).inverse(), mkpts1[inliers], mkpts2[inliers], inliers

