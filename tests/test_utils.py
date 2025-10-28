from dataclasses import dataclass
from pathlib import Path
import random

import gtsam
from registration.registration import StereoCalibration, StereoFrame, FramePairWithGroundTruth
import tartanair as ta
from util import convert_coordinate_frame, se3_flattened_to_pose3, se3_to_pose3
import numpy as np

# setup tartanair
tartanair_data_root = str(Path(__file__).parent / 'data')
ta.init(tartanair_data_root)

# converts the TartanAir coordinate frame to the CV coordinate frame
TA_TO_CV = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])


# https://tartanair.org/modalities.html
tartanair_calib = StereoCalibration.create(
    K=np.array([
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 320.0],
        [0.0, 0.0, 1.0]
    ]),
    T=np.array([0.0, 0.25, 0.0]),
    R=np.eye(3)
)

def get_tartanair_iterator_with_odometry(env="ArchVizTinyHouseDay", difficulty="easy", traj="P000", rotation_noise_sigmas=np.array([np.deg2rad(0.2), np.deg2rad(0.2), np.deg2rad(0.2)]), translation_noise_sigmas=np.array([0.005, 0.005, 0.005]), include_ground_truth=False):
    ta_iterator = ta.iterator(
        env=[env],
        difficulty=difficulty,
        trajectory_id=traj,
        modality=["image"],
        camera_name=["lcam_front", "rcam_front"],
    )

    world_to_prev_robot = None
    world_to_first_robot = None

    for sample in ta_iterator:
        world_to_robot = se3_flattened_to_pose3(sample['lcam_front']['pose'])
        world_to_robot = convert_coordinate_frame(world_to_robot, TA_TO_CV)

        if world_to_prev_robot is None or world_to_first_robot is None:
            world_to_prev_robot = world_to_robot
            world_to_first_robot = world_to_robot
            continue # skip since we can't provide a relative pose having seen only a single pose

        prev_robot_to_robot = world_to_prev_robot.inverse() * world_to_robot

        # add noise to the relative odometry measurement
        rotation_noise = gtsam.Rot3.Expmap(np.random.normal(0, rotation_noise_sigmas))
        translation_noise = gtsam.Point3(np.random.normal(0, translation_noise_sigmas))
        prev_robot_to_robot_noise = gtsam.Pose3(rotation_noise, translation_noise)
        noisy_prev_robot_to_robot = prev_robot_to_robot.compose(prev_robot_to_robot_noise) # should be equivalent to prev_robot_to_robot_noise * prev_robot_to_robot?

        # prepare ground truth pose
        first_robot_to_robot = world_to_first_robot.inverse() * world_to_robot

        # update previous world to robot pose
        world_to_prev_robot = world_to_robot

        # prepare frame
        frame =StereoFrame(
            left=sample['lcam_front']['image'],
            right=sample['rcam_front']['image'],
            calibration=tartanair_calib
        )

        if include_ground_truth:
            yield frame, noisy_prev_robot_to_robot, first_robot_to_robot
        else:
            yield frame, noisy_prev_robot_to_robot


def load_tartanair_pair(env="ArchVizTinyHouseDay", difficulty="easy", traj="P000", max_dist=1.0, max_degs=20.0, seed=0):
    """
    Load a pair of sensor data samples from the TartanAir dataset at two close steps along the trajectory.
    """

    ta_iterator = ta.iterator(
        env=[env],
        difficulty=difficulty,
        trajectory_id=traj,
        modality=["image"],
        camera_name=["lcam_front", "rcam_front"],
    )

    traj_len = list((Path(tartanair_data_root) / env / f"Data_{difficulty}" / traj / "image_lcam_front").glob("*.png")).__len__()

    random.seed(seed)
    start_idx = random.randint(0, traj_len - 2)
    end_idx = start_idx + 1

    for _ in range(start_idx + 1):
        first = next(ta_iterator)
    first_pose = se3_flattened_to_pose3(first['lcam_front']['pose'])
    first_pose = convert_coordinate_frame(first_pose, TA_TO_CV)

    second = next(ta_iterator)
    second_pose = se3_flattened_to_pose3(second['lcam_front']['pose'])
    second_pose = convert_coordinate_frame(second_pose, TA_TO_CV)

    while True:
        idx = end_idx + 1

        try:
            sample = next(ta_iterator)
        except StopIteration:
            break

        sample_pose = se3_flattened_to_pose3(sample['lcam_front']['pose'])
        sample_pose = convert_coordinate_frame(sample_pose, TA_TO_CV)

        sample_dist = np.linalg.norm(sample_pose.translation() - first_pose.translation())
        sample_angle = np.linalg.norm((first_pose.inverse() * sample_pose).rotation().ypr())

        if sample_dist < max_dist and sample_angle < np.deg2rad(max_degs):
            end_idx = idx
            second = sample
            second_pose = sample_pose
        else:
            break

    return FramePairWithGroundTruth[StereoFrame](
        first=StereoFrame(
            left=first['lcam_front']['image'],
            right=first['rcam_front']['image'],
            calibration=tartanair_calib
        ),
        second=StereoFrame(
            left=second['lcam_front']['image'],
            right=second['rcam_front']['image'],
            calibration=tartanair_calib
        ),
        first_T_second=first_pose.inverse() * second_pose
    )

def print_pose_error(*, estimated_pose: gtsam.Pose3 | None = None, ground_truth_pose: gtsam.Pose3 | None = None):
    if estimated_pose is not None and ground_truth_pose is not None:
        pose_error = estimated_pose * ground_truth_pose.inverse()
        
        print('--------------------------------')
        print(f'Estimated translation: ({np.linalg.norm(estimated_pose.translation()):.2f} m) [{estimated_pose.translation()[0]:.2f}, {estimated_pose.translation()[1]:.2f}, {estimated_pose.translation()[2]:.2f}]')
        print(f'Estimated rotation: ({np.linalg.norm(np.rad2deg(estimated_pose.rotation().ypr())):.2f} deg) [{np.rad2deg(estimated_pose.rotation().ypr()[0]):.2f}, {np.rad2deg(estimated_pose.rotation().ypr()[1]):.2f}, {np.rad2deg(estimated_pose.rotation().ypr()[2]):.2f}]')
        print('--------------------------------')
        print(f'Ground truth translation: ({np.linalg.norm(ground_truth_pose.translation()):.2f} m) [{ground_truth_pose.translation()[0]:.2f}, {ground_truth_pose.translation()[1]:.2f}, {ground_truth_pose.translation()[2]:.2f}]')
        print(f'Ground truth rotation: ({np.linalg.norm(np.rad2deg(ground_truth_pose.rotation().ypr())):.2f} deg) [{np.rad2deg(ground_truth_pose.rotation().ypr()[0]):.2f}, {np.rad2deg(ground_truth_pose.rotation().ypr()[1]):.2f}, {np.rad2deg(ground_truth_pose.rotation().ypr()[2]):.2f}]')
        print('--------------------------------')
        print(f'Pose error translation: ({np.linalg.norm(pose_error.translation()):.2f} m) [{pose_error.translation()[0]:.2f}, {pose_error.translation()[1]:.2f}, {pose_error.translation()[2]:.2f}]')
        print(f'Pose error rotation: ({np.linalg.norm(np.rad2deg(pose_error.rotation().ypr())):.2f} deg) [{np.rad2deg(pose_error.rotation().ypr()[0]):.2f}, {np.rad2deg(pose_error.rotation().ypr()[1]):.2f}, {np.rad2deg(pose_error.rotation().ypr()[2]):.2f}]')
        print('--------------------------------')
    elif estimated_pose is not None:
        print('--------------------------------')
        print(f'Estimated translation: ({np.linalg.norm(estimated_pose.translation()):.2f} m) [{estimated_pose.translation()[0]:.2f}, {estimated_pose.translation()[1]:.2f}, {estimated_pose.translation()[2]:.2f}]')
        print(f'Estimated rotation: ({np.linalg.norm(np.rad2deg(estimated_pose.rotation().ypr())):.2f} deg) [{np.rad2deg(estimated_pose.rotation().ypr()[0]):.2f}, {np.rad2deg(estimated_pose.rotation().ypr()[1]):.2f}, {np.rad2deg(estimated_pose.rotation().ypr()[2]):.2f}]')
        print('--------------------------------')
    elif ground_truth_pose is not None:
        print('--------------------------------')
        print(f'Ground truth translation: ({np.linalg.norm(ground_truth_pose.translation()):.2f} m) [{ground_truth_pose.translation()[0]:.2f}, {ground_truth_pose.translation()[1]:.2f}, {ground_truth_pose.translation()[2]:.2f}]')
        print(f'Ground truth rotation: ({np.linalg.norm(np.rad2deg(ground_truth_pose.rotation().ypr())):.2f} deg) [{np.rad2deg(ground_truth_pose.rotation().ypr()[0]):.2f}, {np.rad2deg(ground_truth_pose.rotation().ypr()[1]):.2f}, {np.rad2deg(ground_truth_pose.rotation().ypr()[2]):.2f}]')
        print('--------------------------------')
    else:
        raise ValueError('At least one of estimated_pose or ground_truth_pose must be provided')
