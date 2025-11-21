import gtsam
import rerun as rr
from registration.registration import RectifiedStereoFrame, StereoDepthFrame


def rr_log_pose(
    path: str,
    pose: gtsam.Pose3,
    frame: RectifiedStereoFrame | StereoDepthFrame,
    camera_xyz: rr.ViewCoordinates = rr.ViewCoordinates.RDF,
    image_plane_dist: float = 0.2,
) -> None:
    rr.log(path, rr.Transform3D(translation=pose.translation(), quaternion=pose.rotation().toQuaternion().coeffs()))
    rr.log(path + "/rgb", rr.Image(frame.left_rect).compress())
    rr.log(
        path + "/rgb",
        rr.Pinhole(
            camera_xyz=camera_xyz,
            focal_length=[frame.calibration.K_left_rect[0, 0], frame.calibration.K_left_rect[1, 1]],
            principal_point=[frame.calibration.K_left_rect[0, 2], frame.calibration.K_left_rect[1, 2]],
            image_plane_distance=image_plane_dist,
        ),
    )

    rr_log_pose_arrows(path + "/pose", pose, arrow_length=image_plane_dist / 2)
    rr.log(path + "/position", rr.Points3D([[0, 0, 0]], colors=[[0, 0, 255]], radii=image_plane_dist * 0.1))

    if isinstance(frame, StereoDepthFrame):
        rr.log(path + "/depth", rr.DepthImage(frame.left_depth))
        rr.log(
            path + "/depth",
            rr.Pinhole(
                camera_xyz=camera_xyz,
                focal_length=[frame.calibration.K_left_rect[0, 0], frame.calibration.K_left_rect[1, 1]],
                principal_point=[frame.calibration.K_left_rect[0, 2], frame.calibration.K_left_rect[1, 2]],
                image_plane_distance=image_plane_dist,
            ),
        )


def rr_log_pose_arrows(path: str, pose: gtsam.Pose3, arrow_length: float = 0.2) -> None:
    rr.log(
        path + "/pose",
        rr.Arrows3D(
            vectors=[[arrow_length, 0.0, 0.0], [0.0, arrow_length, 0.0], [0.0, 0.0, arrow_length]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ),
    )


def rr_log_trajectory(
    path: str,
    trajectory: list[gtsam.Pose3],
    color: tuple[int, int, int] = (0, 0, 255),
    radii: float = 0.1,
) -> None:
    if len(trajectory) < 2:
        return
    strips = []
    for i in range(len(trajectory) - 1):
        strips.append([trajectory[i].translation(), trajectory[i + 1].translation()])

    rr.log(path, rr.LineStrips3D(strips=strips, colors=color, radii=radii))
