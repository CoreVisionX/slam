from dataclasses import dataclass
from pathlib import Path
import random

import gtsam
from registration.registration import StereoCalibration, StereoFrame, StereoPairWithGroundTruth
import tartanair as ta
from util import se3_flattened_to_pose3, se3_to_pose3
import numpy as np

# setup tartanair
tartanair_data_root = str(Path(__file__).parent / 'data')
ta.init(tartanair_data_root)

# https://tartanair.org/modalities.html
tartanair_calib = StereoCalibration.create(
    K=np.array([
        [0.32, 0.0, 0.32],
        [0.0, 0.32, 0.32],
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

    for _ in range(start_idx + 1):
        first = next(ta_iterator)
    first_pose = se3_flattened_to_pose3(first['lcam_front']['pose'])

    second = next(ta_iterator)

    while True:
        idx = end_idx + 1

        try:
            sample = next(ta_iterator)
        except StopIteration:
            break

        sample_pose = se3_flattened_to_pose3(sample['lcam_front']['pose'])

        sample_dist = np.linalg.norm(sample_pose.translation() - first_pose.translation())
        sample_angle = (sample_pose.rotation() * first_pose.rotation().inverse()).axisAngle()[1]

        if sample_dist < max_dist and sample_angle < np.deg2rad(max_degs):
            end_idx = idx
            second = sample
            second_pose = sample_pose
        else:
            break

    # TODO: add calibration

    return StereoPairWithGroundTruth[StereoFrame](
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