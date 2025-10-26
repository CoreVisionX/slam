import cv2
import numpy as np
import torch
import kornia as K

from registration.registration import FramePair, RectifiedStereoFrame, StereoFrame

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


def fundamental_fitler(mkpts1: np.ndarray, mkpts2: np.ndarray) -> np.ndarray:
    Fm, inliers = cv2.findFundamentalMat(
        mkpts1, mkpts2, cv2.USAC_MAGSAC, 1.0, 0.999, 100000
    )
    mask = inliers.ravel() > 0

    return mkpts1[mask], mkpts2[mask], mask


def solve_pnp():
    ...

