# %%
from dataclasses import asdict
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from registration.lightglue import LightglueMatcher
from registration.registration import FramePairWithGroundTruth, StereoDepthFrame
from registration.utils import fundamental_fitler, rectify_stereo_frame_pair, solve_pnp, stack_pair_images, draw_matches
import tests.test_utils


pair, first_depth, second_depth = tests.test_utils.load_tartanair_pair(seed=10, max_degs=30.0, max_dist=2.0, traj='P003')

plt.imshow(stack_pair_images(pair, 'left'))

print(f'Translation: ({np.linalg.norm(pair.first_T_second.translation()):.2f} m) [{pair.first_T_second.translation()[0]:.2f}, {pair.first_T_second.translation()[1]:.2f}, {pair.first_T_second.translation()[2]:.2f}]')
print(f'Rotation: ({np.linalg.norm(np.rad2deg(pair.first_T_second.rotation().ypr())):.2f} deg) [{np.rad2deg(pair.first_T_second.rotation().ypr()[0]):.2f}, {np.rad2deg(pair.first_T_second.rotation().ypr()[1]):.2f}, {np.rad2deg(pair.first_T_second.rotation().ypr()[2]):.2f}]')

# %%
rectified_pair = rectify_stereo_frame_pair(pair)

plt.imshow(stack_pair_images(rectified_pair, 'left_rect'))

print('Rectified K:', rectified_pair.first.calibration.K_left_rect)

# %%
matcher = LightglueMatcher(num_features=4096)
mkpts1, mkpts2 = matcher.match(rectified_pair)
print('mkpts1', mkpts1.shape)
print('mkpts2', mkpts2.shape)

filtered_mkpts1, filtered_mkpts2, inlier_mask = fundamental_fitler(mkpts1, mkpts2)

plt.imshow(draw_matches(rectified_pair, filtered_mkpts1[::32], filtered_mkpts2[::32], inlier_mask[::32]))

# %%
max_depth = 10.0
first_depth[first_depth > max_depth] = np.nan
second_depth[second_depth > max_depth] = np.nan

plt.imshow(first_depth)
plt.colorbar()
plt.show()

print('Mean depth:', np.nanmean(first_depth))

# %%

def depth_to_3d(depth_image, intrinsic_matrix):
    """
    Converts a depth image to a 3D point cloud using camera intrinsics.

    Args:
        depth_image: A NumPy array of shape (H, W) representing the depth map.
        intrinsic_matrix: The camera's intrinsic matrix (K).

    Returns:
        A NumPy array of shape (H, W, 3) representing 3D points in camera coordinates.
    """
    height, width = depth_image.shape
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Compute normalized image coordinates
    X = (u - cx) / fx * depth_image
    Y = (v - cy) / fy * depth_image
    Z = depth_image

    # Stack into (H, W, 3)
    points_3d = np.stack((X, Y, Z), axis=-1)

    return points_3d

depth_pair = FramePairWithGroundTruth(
    first=StereoDepthFrame(
        left=rectified_pair.first.left,
        right=rectified_pair.first.right,
        calibration=rectified_pair.first.calibration,
        left_rect=rectified_pair.first.left_rect,
        right_rect=rectified_pair.first.right_rect,
        left_depth=first_depth,
        left_depth_xyz=depth_to_3d(first_depth, rectified_pair.first.calibration.K_left_rect)
    ),
    second=StereoDepthFrame(
        left=rectified_pair.second.left,
        right=rectified_pair.second.right,
        calibration=rectified_pair.second.calibration,
        left_rect=rectified_pair.second.left_rect,
        right_rect=rectified_pair.second.right_rect,
        left_depth=second_depth,
        left_depth_xyz=depth_to_3d(second_depth, rectified_pair.second.calibration.K_left_rect)
    ),
    first_T_second=pair.first_T_second
)

first_T_second, pnp_filtered_mkpts1, pnp_filtered_mkpts2, pnp_inliers = solve_pnp(depth_pair, filtered_mkpts1, filtered_mkpts2)

pose_error = first_T_second * pair.first_T_second.inverse()

plt.imshow(draw_matches(depth_pair, pnp_filtered_mkpts1[::4], pnp_filtered_mkpts2[::4]))

# TODO: factor out printing utilities
# remember, composition over inheritance. Don't write the exact same code a bunch of times.
print('--------------------------------')
print(f'Estimated translation: ({np.linalg.norm(first_T_second.translation()):.2f} m) [{first_T_second.translation()[0]:.2f}, {first_T_second.translation()[1]:.2f}, {first_T_second.translation()[2]:.2f}]')
print(f'Estimated rotation: ({np.linalg.norm(np.rad2deg(first_T_second.rotation().ypr())):.2f} deg) [{np.rad2deg(first_T_second.rotation().ypr()[0]):.2f}, {np.rad2deg(first_T_second.rotation().ypr()[1]):.2f}, {np.rad2deg(first_T_second.rotation().ypr()[2]):.2f}]')
print('--------------------------------')
print(f'Ground truth translation: ({np.linalg.norm(pair.first_T_second.translation()):.2f} m) [{pair.first_T_second.translation()[0]:.2f}, {pair.first_T_second.translation()[1]:.2f}, {pair.first_T_second.translation()[2]:.2f}]')
print(f'Ground truth rotation: ({np.linalg.norm(np.rad2deg(pair.first_T_second.rotation().ypr())):.2f} deg) [{np.rad2deg(pair.first_T_second.rotation().ypr()[0]):.2f}, {np.rad2deg(pair.first_T_second.rotation().ypr()[1]):.2f}, {np.rad2deg(pair.first_T_second.rotation().ypr()[2]):.2f}]')
print('--------------------------------')
print(f'Pose error translation: ({np.linalg.norm(pose_error.translation()):.2f} m) [{pose_error.translation()[0]:.2f}, {pose_error.translation()[1]:.2f}, {pose_error.translation()[2]:.2f}]')
print(f'Pose error rotation: ({np.linalg.norm(np.rad2deg(pose_error.rotation().ypr())):.2f} deg) [{np.rad2deg(pose_error.rotation().ypr()[0]):.2f}, {np.rad2deg(pose_error.rotation().ypr()[1]):.2f}, {np.rad2deg(pose_error.rotation().ypr()[2]):.2f}]')
print('--------------------------------')


# %%
def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    if H is None or mask is None:
        # Fallback: just return the current image if we can’t estimate H
        return img2.copy(), None
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # solid green

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 5) for p in dst_points]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches, mask

img_matches, mask = warp_corners_and_draw_matches(pnp_filtered_mkpts1, pnp_filtered_mkpts2, rectified_pair.first.left_rect, rectified_pair.second.left_rect)
plt.imshow(img_matches)
plt.show()

# %%