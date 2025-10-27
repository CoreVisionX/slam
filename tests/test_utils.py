from dataclasses import dataclass
from pathlib import Path
import random

import gtsam
from registration.registration import StereoCalibration, StereoFrame, FramePairWithGroundTruth
import tartanair as ta
from util import se3_flattened_to_pose3, se3_to_pose3
import numpy as np

# setup tartanair
tartanair_data_root = str(Path(__file__).parent / 'data')
ta.init(tartanair_data_root)

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

def load_tartanair_pair(env="ArchVizTinyHouseDay", difficulty="easy", traj="P000", max_dist=1.0, max_degs=20.0, seed=0):
    """
    Load a pair of sensor data samples from the TartanAir dataset at two close steps along the trajectory.
    """

    ta_iterator = ta.iterator(
        env=[env],
        difficulty=difficulty,
        trajectory_id=traj,
        modality=["image", "depth"],
        camera_name=["lcam_front", "rcam_front"],
    )

    traj_len = list((Path(tartanair_data_root) / env / f"Data_{difficulty}" / traj / "image_lcam_front").glob("*.png")).__len__()

    random.seed(seed)
    start_idx = random.randint(0, traj_len - 2)
    end_idx = start_idx + 1

    TA_TO_CV = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])

    for _ in range(start_idx + 1):
        first = next(ta_iterator)
    first_pose = se3_flattened_to_pose3(first['lcam_front']['pose'])

    first_pose_R = TA_TO_CV @ first_pose.rotation().matrix() @ TA_TO_CV.T
    first_pose_t = TA_TO_CV @ first_pose.translation()
    first_pose = gtsam.Pose3(gtsam.Rot3(first_pose_R), gtsam.Point3(first_pose_t))

    second = next(ta_iterator)
    second_pose = se3_flattened_to_pose3(second['lcam_front']['pose'])

    second_pose_R = TA_TO_CV @ second_pose.rotation().matrix() @ TA_TO_CV.T
    second_pose_t = TA_TO_CV @ second_pose.translation()
    second_pose = gtsam.Pose3(gtsam.Rot3(second_pose_R), gtsam.Point3(second_pose_t))

    while True:
        idx = end_idx + 1

        try:
            sample = next(ta_iterator)
        except StopIteration:
            break

        sample_pose = se3_flattened_to_pose3(sample['lcam_front']['pose'])

        sample_pose_R = TA_TO_CV @ sample_pose.rotation().matrix() @ TA_TO_CV.T
        sample_pose_t = TA_TO_CV @ sample_pose.translation()
        sample_pose = gtsam.Pose3(gtsam.Rot3(sample_pose_R), gtsam.Point3(sample_pose_t))

        sample_dist = np.linalg.norm(sample_pose.translation() - first_pose.translation())
        sample_angle = (sample_pose.rotation() * first_pose.rotation().inverse()).axisAngle()[1]

        if sample_dist < max_dist and sample_angle < np.deg2rad(max_degs):
            end_idx = idx
            second = sample
            second_pose = sample_pose
        else:
            break

    # TODO: add calibration

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
    ), first['lcam_front']['depth'], second['lcam_front']['depth']