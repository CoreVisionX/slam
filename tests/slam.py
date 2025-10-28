# %%
import sys
import os
import gtsam
import matplotlib.pyplot as plt
import rerun as rr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_utils import get_tartanair_iterator_with_odometry
from viz import rr_log_pose, rr_log_trajectory

odometry_iterator = get_tartanair_iterator_with_odometry(env='AbandonedFactory', difficulty='easy', traj='P001', include_ground_truth=True)
rr.init("slam", spawn=True)

trajectory: list[gtsam.Pose3] = []
gt_trajectory: list[gtsam.Pose3] = []

for i, (frame, noisy_prev_robot_to_robot, first_robot_to_robot) in enumerate(odometry_iterator):
    rectified_frame = frame.rectify()
    gt_trajectory.append(first_robot_to_robot)

    if len(trajectory) == 0:
        trajectory.append(noisy_prev_robot_to_robot)
    else:
        trajectory.append(trajectory[-1].compose(noisy_prev_robot_to_robot))

    rr.set_time("frame", sequence=i)

    rr_log_trajectory("gt_trajectory", gt_trajectory, color=(0, 255, 0))
    rr_log_pose("gt", first_robot_to_robot, rectified_frame)

    rr_log_trajectory("trajectory", trajectory, color=(0, 0, 255))
    rr_log_pose("pose", trajectory[-1], rectified_frame)

# %%
