# %%
import sys
import os
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from registration.lightglue import LightglueMatcher
from registration.utils import fundamental_fitler, rectify_stereo_frame_pair, stack_pair_images, draw_matches
import tests.test_utils

pair = tests.test_utils.load_tartanair_pair()

plt.imshow(stack_pair_images(pair, 'left'))

print('Translation:', pair.first_T_second.translation())
print('Rotation:', np.rad2deg(pair.first_T_second.rotation().ypr()))
print(np.rad2deg(pair.first_T_second.rotation().ypr()))

# %%
rectified_pair = rectify_stereo_frame_pair(pair)

plt.imshow(stack_pair_images(rectified_pair, 'left_rect'))

print('Rectified K:', rectified_pair.first.calibration.K_left_rect)

# %%
matcher = LightglueMatcher(num_features=2048)
mkpts1, mkpts2 = matcher.match(rectified_pair)

filtered_mkpts1, filtered_mkpts2, inlier_mask = fundamental_fitler(mkpts1, mkpts2)

plt.imshow(draw_matches(rectified_pair, filtered_mkpts1[::32], filtered_mkpts2[::32], inlier_mask[::32]))
# %%
