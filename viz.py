from registration.registration import RectifiedStereoFrame, StereoDepthFrame
import rerun as rr
import gtsam
import numpy as np

def rr_log_pose(path: str, pose: gtsam.Pose3, frame: RectifiedStereoFrame | StereoDepthFrame, camera_xyz: rr.ViewCoordinates = rr.ViewCoordinates.RDF):
    rr.log(path, rr.Transform3D(translation=pose.translation(), quaternion=pose.rotation().toQuaternion().coeffs()))
    rr.log(path + "/rgb", rr.Image(frame.left_rect))
    rr.log(path + "/rgb", rr.Pinhole(camera_xyz=camera_xyz, focal_length=[frame.calibration.K_left_rect[0, 0], frame.calibration.K_left_rect[1, 1]], principal_point=[frame.calibration.K_left_rect[0, 2], frame.calibration.K_left_rect[1, 2]]))

    if isinstance(frame, StereoDepthFrame):
        rr.log(path + "/depth", rr.DepthImage(frame.left_depth))
        rr.log(path + "/depth", rr.Pinhole(camera_xyz=camera_xyz, focal_length=[frame.calibration.K_left_rect[0, 0], frame.calibration.K_left_rect[1, 1]], principal_point=[frame.calibration.K_left_rect[0, 2], frame.calibration.K_left_rect[1, 2]]))


def rr_log_trajectory(path: str, trajectory: list[gtsam.Pose3], color: tuple[int, int, int] = (0, 0, 255), radii: float = 0.008):
    strips = []
    for i in range(len(trajectory) - 1):
        strips.append([trajectory[i].translation(), trajectory[i + 1].translation()])

    rr.log(path, rr.LineStrips3D(strips=strips, colors=color, radii=radii))
