import gtsam
import numpy as np


def se3_to_pose3(se3: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(se3[:3, :3]), gtsam.Point3(se3[:3, 3]))

def se3_flattened_to_pose3(se3_flattened: np.ndarray) -> gtsam.Pose3:
    T = np.eye(4)
    T[:3, :3] = se3_flattened[:9].reshape(3, 3)
    T[:3, 3] = se3_flattened[9:]

    return se3_to_pose3(T)

