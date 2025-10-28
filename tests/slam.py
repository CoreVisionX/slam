# %%
import sys
import os
import time
import gtsam
from gtsam.symbol_shorthand import X
import numpy as np
import rerun as rr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registration.registration import FramePair, StereoDepthFrame
from backend.pose_graph import GtsamPoseGraph
from backend.proximity_loop_detector import ProximityLoopDetector
from tests.test_utils import get_tartanair_iterator_with_odometry, tartanair_calib
from depth.sgbm import SGBM
from registration.lightglue import LightglueMatcher
from registration.utils import fundamental_fitler, solve_pnp
from viz import rr_log_pose, rr_log_trajectory, rr_log_graph_edges

odometry_iterator = get_tartanair_iterator_with_odometry(
    env='AbandonedFactory', 
    difficulty='easy', 
    traj='P001', 
    include_ground_truth=True,
    rotation_noise_sigmas=np.array([np.deg2rad(0.5), np.deg2rad(0.5), np.deg2rad(0.5)]),
    translation_noise_sigmas=np.array([0.008, 0.008, 0.008])
)

# TODO: the noise should probably be much higher than the real noise since we're integrating the odometry poses between keyframes
odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.deg2rad(0.2), np.deg2rad(0.2), np.deg2rad(0.2), 0.005, 0.005, 0.005]))

rr.init("slam", spawn=True)

sgbm = SGBM(num_disparities=16 * 6, block_size=5, image_color='RGB')

# setup backend
pose_graph = GtsamPoseGraph(K=tartanair_calib.K_left_rect)
loop_detector = ProximityLoopDetector(max_translation=5.0, max_rotation=np.deg2rad(35), max_candidates=5, min_seperation=1)

# loop_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.deg2rad(0.5), np.deg2rad(0.5), np.deg2rad(0.5), 0.01, 0.01, 0.01]))
loop_noise = odometry_noise # set it to the odometry noise so they're weighted equally which will make debugging the influence of the loop closures easier

# TODO: consider factoring this out into it's own class too
# TODO: log match images somehow?
# TODO: log numbers and changes in inlier counts somehow?
# setup two-view pose estimation
matcher = LightglueMatcher(num_features=2048)
min_inlier_count = 200

# TODO: refactor keyframing logic into it's own class
keyframe_translation_threshold = 1.0
keyframe_rotation_threshold = np.deg2rad(20)
cur_keyframe_pose = None

raw_trajectory: list[gtsam.Pose3] = []
gt_trajectory: list[gtsam.Pose3] = []

gt_keyframe_trajectory: list[gtsam.Pose3] = []

prev_frame = None

for i, (frame, noisy_prev_robot_to_robot, first_robot_to_robot) in enumerate(odometry_iterator):
    # update raw/gt trajectories
    gt_trajectory.append(first_robot_to_robot)

    if len(raw_trajectory) == 0:
        raw_trajectory.append(noisy_prev_robot_to_robot)
    else:
        raw_trajectory.append(raw_trajectory[-1].compose(noisy_prev_robot_to_robot))

    # preprocess frame
    rectified_frame = frame.rectify()
    depth_frame = sgbm.compute_depth(rectified_frame)

    # update slam if a new keyframe is detected
    if cur_keyframe_pose is None:
        cur_keyframe_pose = first_robot_to_robot
        pose_graph.process_odometry(noisy_prev_robot_to_robot, odometry_noise, depth_frame)

        # TODO: figure out why doing this causes the estimated trajectory to be offset? I think it's because even this is actually the second frame not the first one or something?
        # # TODO: this is a hack to get the loop detector to work with the first keyframe, fix this later
        # pose_graph.frames[0] = depth_frame

    # TODO: setup a proper reusable VO pipeline if you're going to use this
    # # use visual odometry instead of the given raw odometry
    # if prev_frame is not None:
    #     try:
    #         pair = FramePair[StereoDepthFrame](first=prev_frame, second=depth_frame)
    #         mkpts1, mkpts2 = matcher.match(pair)
    #         filtered_mkpts1, filtered_mkpts2, inlier_mask = fundamental_fitler(mkpts1, mkpts2)
    #         first_to_second, pnp_filtered_mkpts1, pnp_filtered_mkpts2, pnp_inliers = solve_pnp(pair, filtered_mkpts1, filtered_mkpts2)

    #         if len(pnp_inliers) < min_inlier_count:
    #             raise ValueError("Not enough final inliers to solve PnP")
           
    #         noisy_prev_robot_to_robot = first_to_second
    #         print(f"Computed visual odometry between {len(pnp_inliers)} inliers")

    #     except ValueError:
    #         print(f"Failed to solve PnP for visual odometry between {prev_frame.idx} and {depth_frame.idx}")

    prev_frame = depth_frame

    # use raw odometry for keyframe calculations
    cur_keyframe_to_robot = cur_keyframe_pose.inverse() * raw_trajectory[-1]
    dist = np.linalg.norm(cur_keyframe_to_robot.translation())
    rot = np.linalg.norm(cur_keyframe_to_robot.rotation().ypr()) # TODO: use axis-angle or quaternions to avoid potential singularity issues?

    if dist > keyframe_translation_threshold or rot > keyframe_rotation_threshold:
        start_time = time.perf_counter()

        gt_keyframe_trajectory.append(first_robot_to_robot)

        # update keyframe pose
        cur_keyframe_pose = raw_trajectory[-1]

        # update slam with new keyframe
        # cur_keyframe_to_robot accumulates all of the odometry since the last keyframe
        pose_graph.process_odometry(cur_keyframe_to_robot, odometry_noise, depth_frame)

        loop_candidates = loop_detector.candidates(pose_graph, pose_graph.kf_idx)
        for candidate in loop_candidates:
            mkpts1, mkpts2 = matcher.match(candidate)

            try:
                filtered_mkpts1, filtered_mkpts2, inlier_mask = fundamental_fitler(mkpts1, mkpts2)
                first_to_second, pnp_filtered_mkpts1, pnp_filtered_mkpts2, pnp_inliers = solve_pnp(candidate, filtered_mkpts1, filtered_mkpts2)
            except Exception as e:
                print(f"Failed to solve PnP for loop closure between {candidate.first_idx} and {candidate.second_idx}: {e}")
                continue

            if len(pnp_inliers) > min_inlier_count:
                pose_graph.add_between_pose_factor(candidate.first_idx, candidate.second_idx, first_to_second, loop_noise)
                print(f"Added loop closure between {candidate.first_idx} and {candidate.second_idx} with {len(pnp_inliers)} inliers")
            else:
                print(f"Failed to add loop closure between {candidate.first_idx} and {candidate.second_idx} with {len(pnp_inliers)} inliers")

            # TODO: log the error the relative to the ground truth relative pose?

        pose_graph.optimize()

        end_time = time.perf_counter()
        print(f"Frame Processing Time: {end_time - start_time:.2f} seconds ({1 / (end_time - start_time):.2f} fps)")
    else:
        continue

    # logging
    rr.set_time("frame", sequence=i)

    rr_log_trajectory("gt_trajectory", gt_trajectory, color=(0, 255, 0))
    rr_log_pose("gt", first_robot_to_robot, depth_frame)

    rr_log_trajectory("raw_trajectory", raw_trajectory, color=(0, 0, 255))
    rr_log_pose("raw", raw_trajectory[-1], depth_frame)

    rr_log_graph_edges(path="graph", nodes=pose_graph.values, graph=pose_graph.graph)
    rr_log_pose(path="optimized", pose=pose_graph.get_latest_pose(), frame=depth_frame)

    trans_ate = np.linalg.norm(gt_keyframe_trajectory[-1].translation() - pose_graph.get_latest_pose().translation())
    rot_ate = np.rad2deg(np.linalg.norm(gt_keyframe_trajectory[-1].rotation().ypr() - pose_graph.get_latest_pose().rotation().ypr()))
    rr.log("/translation/ate", rr.Scalars(trans_ate))
    rr.log("/rotation/ate", rr.Scalars(rot_ate))

# %%
