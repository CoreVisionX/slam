# %%
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registration.lightglue import LightglueMatcher
from registration.utils import fundamental_fitler, rectify_stereo_frame_pair, solve_pnp, stack_pair_images, draw_matches
import tests.test_utils

pair = tests.test_utils.load_tartanair_pair(seed=2, max_degs=40.0, max_dist=5.0, env='AbandonedFactory', traj='P001')

plt.imshow(stack_pair_images(pair, 'left'))

tests.test_utils.print_pose_error(ground_truth_pose=pair.first_T_second)

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
from depth.sgbm import SGBM

sgbm = SGBM(num_disparities=16 * 6, block_size=5, image_color='RGB')
depth_pair = sgbm.compute_depth_pair(rectified_pair, max_depth=30.0)

plt.imshow(depth_pair.first.left_depth)
plt.colorbar()
plt.show()

# %%
first_T_second, pnp_filtered_mkpts1, pnp_filtered_mkpts2, pnp_inliers = solve_pnp(depth_pair, filtered_mkpts1, filtered_mkpts2)

plt.imshow(draw_matches(depth_pair, pnp_filtered_mkpts1[::4], pnp_filtered_mkpts2[::4]))
plt.show()

tests.test_utils.print_pose_error(estimated_pose=first_T_second, ground_truth_pose=pair.first_T_second)
