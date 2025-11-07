import cv2
import gtsam
import numpy as np
import torch
import kornia as K

from registration.registration import FramePair, MatchedFramePair, RectifiedStereoFrame, StereoDepthFrame, FeatureFrame, StereoFrame

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
        mkpts1, mkpts2, cv2.USAC_MAGSAC, 2.0, 0.999, 10_000
    )
    mask = inliers.ravel() > 0

    return mkpts1[mask], mkpts2[mask], mask


def solve_pnp(pair: MatchedFramePair[FeatureFrame]) -> tuple[gtsam.Pose3, MatchedFramePair[FeatureFrame]]:
    # mkpts1 = pair.first.features['keypoints'].cpu().numpy()[pair.matches[:, 0]]
    # mkpts2 = pair.second.features['keypoints'].cpu().numpy()[pair.matches[:, 1]]
    mkpts1, mkpts2 = get_matching_keypoints(
        kp1=pair.first.features['keypoints'].cpu().numpy() if isinstance(pair.first.features['keypoints'], torch.Tensor) else pair.first.features['keypoints'],
        kp2=pair.second.features['keypoints'].cpu().numpy() if isinstance(pair.second.features['keypoints'], torch.Tensor) else pair.second.features['keypoints'],
        idxs=pair.matches
    )

    # u = mkpts1[:, 1].astype(int)
    # v = mkpts1[:, 0].astype(int)

    # mkpts1_3d = pair.first.left_depth_xyz[u, v, :]
    # mkpts1_depth = pair.first.left_depth[u, v]
    # mkpts1_color = pair.first.left[u, v]

    mkpts1_3d = pair.first.features['keypoints_3d'][pair.matches[:, 0]]
    mkpts1_depth = pair.first.features['keypoints_depth'][pair.matches[:, 0]]
    mkpts1_color = pair.first.features['keypoints_color'][pair.matches[:, 0]]

    valid_mask = mkpts1_depth > 0
    mkpts1_3d = mkpts1_3d[valid_mask]
    mkpts1_color = mkpts1_color[valid_mask]
    mkpts1 = mkpts1[valid_mask]
    mkpts2 = mkpts2[valid_mask]

    if mkpts1_3d.shape[0] < 4:
        raise ValueError("Not enough 3D correspondences for PnP")

    obj_points = np.ascontiguousarray(mkpts1_3d.astype(np.float32))
    img_points = np.ascontiguousarray(mkpts2.reshape(-1, 1, 2).astype(np.float32))

    ret, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_points,
        img_points,
        pair.first.calibration.K_left_rect, 
        np.zeros((5, 1)),
        reprojectionError=2.0, # TODO: benchmark reprojectionError. plot error distributions for different reprojectionError values. with a proper config system this should be easy?
        confidence=0.999,
        iterationsCount=100_000,
        flags=cv2.USAC_MAGSAC # TODO: benchmark USAC_MAGSAC vs SOLVEPNP_ITERATIVE
    )
    inliers = inliers.ravel()

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    if not ret:
        raise ValueError("Failed to solve PnP")

    if len(inliers) < 10:
        raise ValueError("Not enough inliers to solve PnP")

    matched_pair = MatchedFramePair(
        first=pair.first,
        second=pair.second,
        matches=pair.matches[valid_mask][inliers]
    )

    return gtsam.Pose3(gtsam.Rot3(R), t).inverse(), matched_pair
