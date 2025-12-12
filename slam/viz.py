import gtsam
import rerun as rr
import numpy as np

from slam.registration.registration import RectifiedStereoFrame, StereoDepthFrame


def log_scalar(path: str, value: float) -> None:
    rr.log(path, rr.Scalars(value))

def rr_log_pose(
    path: str,
    pose: gtsam.Pose3,
    frame: RectifiedStereoFrame | StereoDepthFrame,
    pose_covariance: np.ndarray | None = None,
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

    # if pose_covariance is not None:
    #     # log covariance ellipse around the pose's translation
    #     translation_covariance = pose_covariance[3:, 3:]
        
    #     # for now just use the diagonal of the covariance matrix
    #     std_devs = np.sqrt(np.diag(translation_covariance))
        
    #     rr.log(path + "/pose_covariance", 
    #         rr.Ellipsoids3D(
    #             centers=[[0, 0, 0]],
    #             half_sizes=[std_devs],
    #             colors=[[0, 255, 0]],
    #         )
    #     )

    if pose_covariance is not None:
        # Pose3 tangent ordering in GTSAM: [Rx,Ry,Rz, Tx,Ty,Tz]
        translation_cov = pose_covariance[3:6, 3:6]

        # Eigen-decompose covariance -> ellipsoid axes/orientation
        w, V = np.linalg.eigh(translation_cov)
        w = np.maximum(w, 0.0)

        k = 2.0  # "2-sigma" scale (optional)
        half_sizes = k * np.sqrt(w)  # semi-axis lengths

        # Ensure right-handed basis for a proper rotation
        R_axes = V
        if np.linalg.det(R_axes) < 0:
            R_axes[:, 0] *= -1.0

        def quat_from_rotmat(R: np.ndarray) -> np.ndarray:
            # returns [x, y, z, w]
            t = np.trace(R)
            if t > 0:
                s = np.sqrt(t + 1.0) * 2.0
                qw = 0.25 * s
                qx = (R[2, 1] - R[1, 2]) / s
                qy = (R[0, 2] - R[2, 0]) / s
                qz = (R[1, 0] - R[0, 1]) / s
            else:
                i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
                if i == 0:
                    s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                    qw = (R[2, 1] - R[1, 2]) / s
                    qx = 0.25 * s
                    qy = (R[0, 1] + R[1, 0]) / s
                    qz = (R[0, 2] + R[2, 0]) / s
                elif i == 1:
                    s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                    qw = (R[0, 2] - R[2, 0]) / s
                    qx = (R[0, 1] + R[1, 0]) / s
                    qy = 0.25 * s
                    qz = (R[1, 2] + R[2, 1]) / s
                else:
                    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                    qw = (R[1, 0] - R[0, 1]) / s
                    qx = (R[0, 2] + R[2, 0]) / s
                    qy = (R[1, 2] + R[2, 1]) / s
                    qz = 0.25 * s

            q = np.array([qx, qy, qz, qw], dtype=float)
            return q / np.linalg.norm(q)

        quat_xyzw = quat_from_rotmat(R_axes)

        rr.log(
            path + "/pose_covariance",
            rr.Ellipsoids3D(
                centers=[[0.0, 0.0, 0.0]],
                half_sizes=[half_sizes.tolist()],
                quaternions=[quat_xyzw.tolist()],
                colors=[[0, 255, 0]],
            ),
        )

    # log the RMS translation uncertainty
    if pose_covariance is not None:
        C = pose_covariance[3:6, 3:6]  # translation covariance (Tx,Ty,Tz)

        # Option C: RMS magnitude of translation uncertainty (1σ)
        sigma_mag_rms = float(np.sqrt(np.trace(C)))

        k = 2.0  # if you want a "2σ" magnitude to match your ellipsoid scale
        sigma_mag_rms_k = k * sigma_mag_rms

        rr.log("/pose_covariance/translation_sigma_mag_rms", rr.Scalars(sigma_mag_rms))
        rr.log("/pose_covariance/translation_sigma_mag_rms_2sigma", rr.Scalars(sigma_mag_rms_k))


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
