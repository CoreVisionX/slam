from backend.pose_graph import GtsamPoseGraph
from registration.registration import RectifiedStereoFrame, StereoDepthFrame
import rerun as rr
import gtsam
import numpy as np
import cv2

def rr_log_pose(path: str, pose: gtsam.Pose3, frame: RectifiedStereoFrame | StereoDepthFrame, camera_xyz: rr.ViewCoordinates = rr.ViewCoordinates.RDF, image_plane_dist: float = 0.2):
    rr.log(path, rr.Transform3D(translation=pose.translation(), quaternion=pose.rotation().toQuaternion().coeffs()))
    rr.log(path + "/rgb", rr.Image(frame.left_rect))
    rr.log(path + "/rgb", rr.Pinhole(camera_xyz=camera_xyz, focal_length=[frame.calibration.K_left_rect[0, 0], frame.calibration.K_left_rect[1, 1]], principal_point=[frame.calibration.K_left_rect[0, 2], frame.calibration.K_left_rect[1, 2]], image_plane_distance=image_plane_dist))

    if isinstance(frame, StereoDepthFrame):
        rr.log(path + "/depth", rr.DepthImage(frame.left_depth))
        rr.log(path + "/depth", rr.Pinhole(camera_xyz=camera_xyz, focal_length=[frame.calibration.K_left_rect[0, 0], frame.calibration.K_left_rect[1, 1]], principal_point=[frame.calibration.K_left_rect[0, 2], frame.calibration.K_left_rect[1, 2]], image_plane_distance=image_plane_dist))


def rr_log_trajectory(path: str, trajectory: list[gtsam.Pose3], color: tuple[int, int, int] = (0, 0, 255), radii: float = 0.008):
    strips = []
    for i in range(len(trajectory) - 1):
        strips.append([trajectory[i].translation(), trajectory[i + 1].translation()])

    rr.log(path, rr.LineStrips3D(strips=strips, colors=color, radii=radii))


def rr_log_graph_edges(path: str, nodes, graph):
    loops = []
    scale_free_loops = []
    odom = []

    for idx in range(graph.size()):
        factor = graph.at(idx)
        if not isinstance(factor, gtsam.BetweenFactorPose3) and not isinstance(factor, gtsam.EssentialMatrixConstraint):
            continue

        keys = factor.keys()
        if len(keys) != 2:
            continue

        key_a = int(keys[0])
        key_b = int(keys[1])

        try:
            pose_a = nodes.atPose3(key_a)
            pose_b = nodes.atPose3(key_b)
        except RuntimeError:
            continue

        symbol_a = gtsam.Symbol(key_a)
        symbol_b = gtsam.Symbol(key_b)

        if symbol_a.chr() != symbol_b.chr():
            continue

        if abs(symbol_a.index() - symbol_b.index()) <= 1:
            odom.append([pose_a.translation(), pose_b.translation()])
        else:
            if isinstance(factor, gtsam.EssentialMatrixConstraint):
                scale_free_loops.append([pose_a.translation(), pose_b.translation()])
            else:
                loops.append([pose_a.translation(), pose_b.translation()])

    rr.log(path + "/odom", rr.LineStrips3D(odom, colors=[[255, 0, 0]], radii=[0.004]))
    rr.log(path + "/closures", rr.LineStrips3D(loops, colors=[[128, 128, 0]], radii=[0.004]))
    rr.log(path + "/scale_free_closures", rr.LineStrips3D(scale_free_loops, colors=[[0, 128, 128]], radii=[0.004]))

def rr_log_map_points(path: str, pose_graph: GtsamPoseGraph, points: list[tuple[int, np.ndarray, np.ndarray]], height_colormap: bool = True):
    if len(points) == 0:
        return
        
    world_points = []
    world_points_colors = []

    for kf_idx, pts_3d, pts_color in points:
        assert pts_3d.shape == (len(pts_3d), 3)

        pose = pose_graph.get_pose(kf_idx)
        world_pts_3d = pose.transformFrom(pts_3d.T).T

        world_points.append(world_pts_3d)
        world_points_colors.append(pts_color)
    
    world_points = np.concatenate(world_points, axis=0)
    world_points_colors = np.concatenate(world_points_colors, axis=0)
    assert world_points.shape == (len(world_points), 3)
    assert world_points_colors.shape == (len(world_points_colors), 3)

    if height_colormap:
        height_map = -world_points[:, 1]
        height_normalized = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
        height_colors = cv2.applyColorMap((height_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        world_points_colors = height_colors

    rr.log(path, rr.Points3D(world_points, colors=world_points_colors, radii=[0.01]))
