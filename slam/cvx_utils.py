import gtsam
import numpy as np


def se3_to_pose3(se3: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(se3[:3, :3]), gtsam.Point3(se3[:3, 3]))


def se3_flattened_to_pose3(se3_flattened: np.ndarray) -> gtsam.Pose3:
    se3 = se3_flattened.reshape(3, 4)
    T = np.eye(4)
    T[:3, :3] = se3[:3, :3]
    T[:3, 3] = se3[:3, 3]
    return se3_to_pose3(T)


def convert_coordinate_frame(pose: gtsam.Pose3, old_to_new_frame: np.ndarray) -> gtsam.Pose3:
    new_R = old_to_new_frame @ pose.rotation().matrix() @ old_to_new_frame.T
    new_t = old_to_new_frame @ pose.translation()
    return gtsam.Pose3(gtsam.Rot3(new_R), gtsam.Point3(new_t))

