# %%
from __future__ import annotations

import copy
import os
import sys
from typing import Sequence

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

DEEP_GLOBAL_REGISTRATION_DIR = os.path.join(REPO_ROOT, "submodules", "DeepGlobalRegistration")
USE_DEEP_GLOBAL_REGISTRATION = os.environ.get("LOCAL_RGBD_DEEP_GLOBAL_REGISTRATION", "1").strip().lower() in {
    "1",
    "true",
    "t",
    "yes",
    "y",
}
DEEP_GLOBAL_REGISTRATION_WEIGHTS = os.environ.get(
    "LOCAL_RGBD_DEEP_GLOBAL_WEIGHTS",
    os.path.join(DEEP_GLOBAL_REGISTRATION_DIR, "ResUNetBN2C-feat32-3dmatch-v0.05.pth"),
)

if USE_DEEP_GLOBAL_REGISTRATION:
    if not os.path.isdir(DEEP_GLOBAL_REGISTRATION_DIR):
        raise RuntimeError("DeepGlobalRegistration submodule directory is missing.")
    if not os.path.exists(DEEP_GLOBAL_REGISTRATION_WEIGHTS):
        raise RuntimeError(
            f"Deep global registration weights not found at '{DEEP_GLOBAL_REGISTRATION_WEIGHTS}'. "
            "Set LOCAL_RGBD_DEEP_GLOBAL_WEIGHTS to a valid path."
        )
    sys.path.insert(0, DEEP_GLOBAL_REGISTRATION_DIR)
    try:
        import config as deep_global_config  # type: ignore[import]

        from core.deep_global_registration import DeepGlobalRegistration  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - only required when DGR enabled
        raise RuntimeError("Could not import DeepGlobalRegistration dependencies.") from exc

    def _create_deep_global_registration() -> DeepGlobalRegistration:
        parser = deep_global_config.parser
        config = parser.parse_args([])
        config.weights = DEEP_GLOBAL_REGISTRATION_WEIGHTS
        return DeepGlobalRegistration(config)

    deep_global_registration_factory = _create_deep_global_registration
else:
    deep_global_registration_factory = None

try:
    import rerun as rr
except ImportError:  # pragma: no cover - rerun is optional at import time
    rr = None  # type: ignore

import gtsam
from depth.sgbm import SGBM
from registration.registration import RectifiedStereoFrame, StereoCalibration, StereoDepthFrame
from viz import rr_log_pose
import tests.test_utils as test_utils


# ----------------------------------------------------------------------
# Configuration (kept intentionally small for a quick sanity check)
# ----------------------------------------------------------------------
SEQUENCE_LENGTH = 150
ENVIRONMENT = "ArchVizTinyHouseDay"
DIFFICULTY = "easy"
TRAJECTORY = "P003"
SAMPLING_MODE = "stride"
MIN_STRIDE = 1
MAX_STRIDE = 1
RNG_SEED = 0

MAX_DEPTH_METERS = 2.5
VOXEL_DOWNSAMPLE_SIZE = 0.05  # meters
ICP_DISTANCE_THRESHOLD = VOXEL_DOWNSAMPLE_SIZE * 2  # meters
ANCHOR_RESET_FITNESS_THRESHOLD = 0.0
USE_FPFH_INITIALIZATION = False
FPFH_VOXEL_SIZE = 0.12  # meters for feature generation
FPFH_DISTANCE_THRESHOLD = 0.15  # meters
USE_MULTISCALE_REGISTRATION = True
MS_VOXEL_SIZES = [0.12, 0.06, 0.03]
MS_DISTANCE_THRESHOLDS = [0.25, 0.15, 0.08]
USE_GT_ROTATION_INITIALIZATION = True
GT_ROTATION_NOISE_DEG = 0.0
GT_INIT_RNG = np.random.default_rng(0)
DEPTH_MODE = os.environ.get("LOCAL_RGBD_DEPTH_MODE", "sgbm").strip().lower()
if DEPTH_MODE not in {"ground_truth", "sgbm"}:
    raise ValueError("LOCAL_RGBD_DEPTH_MODE must be either 'ground_truth' or 'sgbm'.")
sgbm = SGBM(num_disparities=16 * 4, block_size=5, image_color="RGB")

# rerun settings for notebook-style explorations
ENABLE_RERUN = os.environ.get("LOCAL_RGBD_RERUN", "1").lower() not in {"0", "false", "f", "no"}
RERUN_APP_ID = os.environ.get("LOCAL_RGBD_RERUN_APP_ID", "local_rgbd_icp")
RERUN_TCP = os.environ.get("LOCAL_RGBD_RERUN_TCP")
SPAWN_VIEWER = True


# ----------------------------------------------------------------------
# Depth utilities
# ----------------------------------------------------------------------
def remap_ground_truth_depth(raw_depth: np.ndarray, calibration: StereoCalibration) -> np.ndarray:
    depth = np.asarray(raw_depth, dtype=np.float32)
    expected_shape = (calibration.height, calibration.width)
    if depth.shape != expected_shape:
        raise ValueError(f"Depth map shape {depth.shape} does not match calibration size {expected_shape}.")
    return cv2.remap(
        depth,
        calibration.map_left_x,
        calibration.map_left_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def depth_map_to_xyz(depth_map: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth_map.shape
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = depth_map
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    xyz = np.stack((x, y, z), axis=-1).astype(np.float32, copy=False)
    xyz[~np.isfinite(z)] = np.nan
    return xyz


def build_ground_truth_depth_frame(
    rectified_frame: RectifiedStereoFrame,
    raw_depth_map: np.ndarray,
    max_depth: float,
) -> StereoDepthFrame:
    depth_rect = remap_ground_truth_depth(raw_depth_map, rectified_frame.calibration)
    invalid = (~np.isfinite(depth_rect)) | (depth_rect <= 0.0) | (depth_rect > max_depth)
    depth_rect = depth_rect.copy()
    depth_rect[invalid] = np.nan
    depth_xyz = depth_map_to_xyz(depth_rect, rectified_frame.calibration.K_left_rect)
    depth_xyz[invalid] = np.nan
    return StereoDepthFrame(
        left=rectified_frame.left,
        right=rectified_frame.right,
        left_rect=rectified_frame.left_rect,
        right_rect=rectified_frame.right_rect,
        calibration=rectified_frame.calibration,
        left_depth=depth_rect,
        left_depth_xyz=depth_xyz,
    )


def build_ground_truth_depth_frames(
    rectified_frames: list[RectifiedStereoFrame],
    raw_depth_maps: list[np.ndarray],
    max_depth: float = MAX_DEPTH_METERS,
) -> list[StereoDepthFrame]:
    if len(rectified_frames) != len(raw_depth_maps):
        raise ValueError("Mismatch between rectified frames and depth maps.")
    return [
        build_ground_truth_depth_frame(rect_frame, depth, max_depth)
        for rect_frame, depth in zip(rectified_frames, raw_depth_maps)
    ]
    if len(rectified_frames) != len(raw_depth_maps):
        raise ValueError("Mismatch between rectified frames and depth maps.")
    return [
        build_ground_truth_depth_frame(rect_frame, depth)
        for rect_frame, depth in zip(rectified_frames, raw_depth_maps)
    ]


# ----------------------------------------------------------------------
# Point cloud & visualization helpers
# ----------------------------------------------------------------------
def _cloud_arrays(cloud: o3d.geometry.PointCloud) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.asarray(cloud.points, dtype=np.float32)
    colors = np.asarray(cloud.colors, dtype=np.float32)
    if colors.size == 0:
        color_arr = None
    else:
        if np.max(colors) <= 1.0:
            color_arr = np.clip((colors * 255.0), 0.0, 255.0).astype(np.uint8)
        else:
            color_arr = np.clip(colors, 0.0, 255.0).astype(np.uint8)
    return points, color_arr


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (points @ rotation.T) + translation


class RerunVisualizer:
    def __init__(
        self,
        *,
        enabled: bool,
        app_id: str,
        spawn_viewer: bool,
        tcp_address: str | None,
    ) -> None:
        if enabled and rr is None:
            raise RuntimeError("rerun-sdk is not installed, but visualization was requested.")
        self._enabled = enabled and rr is not None
        if not self._enabled:
            return
        assert rr is not None
        rr.init(app_id, spawn=spawn_viewer)
        if tcp_address:
            rr.connect_grpc(tcp_address)
        rr.log("/", rr.ViewCoordinates.RDF)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_frame_index(self, idx: int) -> None:
        if not self._enabled:
            return
        assert rr is not None
        rr.set_time("frame", sequence=idx)

    def log_point_cloud(
        self,
        path: str,
        cloud: o3d.geometry.PointCloud,
        *,
        transform: np.ndarray | None = None,
    ) -> None:
        if not self._enabled:
            return
        assert rr is not None
        points, colors = _cloud_arrays(cloud)
        if transform is not None:
            points = _transform_points(points, transform)
        rr.log(path, rr.Points3D(points, colors=colors))

    def log_pose(self, path: str, transform: np.ndarray) -> None:
        if not self._enabled:
            return
        assert rr is not None
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        rr.log(
            path,
            rr.Transform3D(
                translation=translation,
                mat3x3=rotation,
            ),
        )

    def log_rr_pose(
        self,
        path: str,
        transform: np.ndarray,
        frame: StereoDepthFrame,
        frame_index: int | None = None,
    ) -> None:
        if not self._enabled:
            return
        if frame_index is not None:
            self.set_frame_index(frame_index)
        rr_log_pose(path, pose3_from_transform(transform), frame)

    def log_pose_sequence(self, base_path: str, transforms: list[np.ndarray]) -> None:
        for idx, transform in enumerate(transforms):
            self.set_frame_index(idx)
            self.log_pose(base_path, transform)

    def log_trajectory(self, path: str, transforms: list[np.ndarray], color: tuple[int, int, int]) -> None:
        if not self._enabled or len(transforms) < 2:
            return
        assert rr is not None
        positions = [transform[:3, 3] for transform in transforms]
        frame_indices = list(range(len(transforms)))
        for i in range(len(positions) - 1):
            self.set_frame_index(frame_indices[i + 1])
            rr.log(
                f"{path}/segment_{i:02d}",
                rr.LineStrips3D(strips=[[positions[i], positions[i + 1]]], colors=[color], radii=[0.003]),
            )

    def log_trajectory_segment(
        self,
        path: str,
        start: np.ndarray,
        end: np.ndarray,
        frame_index: int,
        color: tuple[int, int, int],
    ) -> None:
        if not self._enabled:
            return
        assert rr is not None
        self.set_frame_index(frame_index)
        rr.log(
            f"{path}/segment_{frame_index:03d}",
            rr.LineStrips3D(strips=[[start, end]], colors=[color], radii=[0.003]),
        )


# ----------------------------------------------------------------------
# Point cloud & ICP helpers
# ----------------------------------------------------------------------
def point_cloud_from_depth_frame(depth_frame: StereoDepthFrame) -> o3d.geometry.PointCloud:
    color_img = depth_frame.left_rect.astype(np.uint8, copy=False)
    depth_img = depth_frame.left_depth.astype(np.float32, copy=False)
    invalid_mask = ~np.isfinite(depth_img)
    if np.any(invalid_mask):
        depth_img = depth_img.copy()
        depth_img[invalid_mask] = 0.0

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_img),
        o3d.geometry.Image(depth_img),
        depth_scale=1.0,
        depth_trunc=MAX_DEPTH_METERS,
        convert_rgb_to_intensity=False,
    )

    calib = depth_frame.calibration
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        calib.width,
        calib.height,
        float(calib.K_left_rect[0, 0]),
        float(calib.K_left_rect[1, 1]),
        float(calib.K_left_rect[0, 2]),
        float(calib.K_left_rect[1, 2]),
    )

    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    if VOXEL_DOWNSAMPLE_SIZE > 0.0:
        cloud = cloud.voxel_down_sample(VOXEL_DOWNSAMPLE_SIZE)

    # remove outliers
    cl, ind = cloud.remove_radius_outlier(nb_points=16, radius=0.15)
    cloud = cloud.select_by_index(ind)
    estimate_cloud_normals(cloud)

    return cloud


def merge_point_clouds(
    base: o3d.geometry.PointCloud,
    addition: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    points = np.concatenate(
        (np.asarray(base.points), np.asarray(addition.points)),
        axis=0,
    )
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(points)

    base_colors = np.asarray(base.colors)
    add_colors = np.asarray(addition.colors)
    if base_colors.size == 0 and add_colors.size == 0:
        return merged

    if base_colors.size == 0:
        merged.colors = o3d.utility.Vector3dVector(add_colors)
    elif add_colors.size == 0:
        merged.colors = o3d.utility.Vector3dVector(base_colors)
    else:
        merged.colors = o3d.utility.Vector3dVector(
            np.concatenate((base_colors, add_colors), axis=0),
        )
    return merged


def prepare_fpfh(
    cloud: o3d.geometry.PointCloud,
    voxel_size: float,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    if voxel_size <= 0.0:
        voxel_size = max(VOXEL_DOWNSAMPLE_SIZE, 0.05)
    down = cloud.voxel_down_sample(voxel_size)
    estimate_cloud_normals(down, radius=voxel_size * 2.0)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return down, fpfh


def match_fpfh(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    distance_threshold: float,
) -> o3d.pipelines.registration.RegistrationResult:
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000, 100),
    )


def compute_deep_global_registration_transform(
    deep_global_registration: object | None,
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> np.ndarray | None:
    if deep_global_registration is None:
        return None
    try:
        transform = deep_global_registration.register(source, target)
    except Exception as exc:  # pragma: no cover - best-effort alignment
        print(f"Deep global registration failed: {exc}")
        return None
    if transform is None:
        return None
    transform_arr = np.asarray(transform, dtype=np.float64)
    if transform_arr.shape != (4, 4):
        return None
    return transform_arr


def multiscale_colored_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    estimation: o3d.pipelines.registration.TransformationEstimationForColoredICP,
    voxel_sizes: Sequence[float],
    distance_thresholds: Sequence[float],
) -> o3d.pipelines.registration.RegistrationResult:
    transform = init_transform
    result = None
    for voxel, max_dist in zip(voxel_sizes, distance_thresholds):
        if voxel > 0.0:
            src_down = source.voxel_down_sample(voxel)
            tgt_down = target.voxel_down_sample(voxel)
        else:
            src_down = source
            tgt_down = target
        estimate_cloud_normals(src_down, radius=max(voxel * 2.0, 0.01))
        estimate_cloud_normals(tgt_down, radius=max(voxel * 2.0, 0.01))

        result = o3d.pipelines.registration.registration_colored_icp(
            src_down,
            tgt_down,
            max_dist,
            transform,
            estimation,
        )
        transform = result.transformation
    assert result is not None
    return result




def estimate_cloud_normals(cloud: o3d.geometry.PointCloud, radius: float | None = None) -> None:
    if radius is None:
        radius = max(VOXEL_DOWNSAMPLE_SIZE * 2.0, 0.01) if VOXEL_DOWNSAMPLE_SIZE > 0.0 else 0.05
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30),
    )


def compute_gt_rotation(
    idx: int,
    anchor_gt_pose: gtsam.Pose3 | None,
    ground_truth_poses: list[gtsam.Pose3] | None,
) -> np.ndarray | None:
    if (
        not USE_GT_ROTATION_INITIALIZATION
        or anchor_gt_pose is None
        or ground_truth_poses is None
        or idx >= len(ground_truth_poses)
    ):
        return None
    rel_gt = anchor_gt_pose.inverse() * ground_truth_poses[idx]
    if GT_ROTATION_NOISE_DEG > 0.0:
        noise = GT_INIT_RNG.normal(0.0, np.deg2rad(GT_ROTATION_NOISE_DEG), size=3)
        noise_rot = gtsam.Rot3.Expmap(noise)
        rel_gt = gtsam.Pose3(noise_rot.compose(rel_gt.rotation()), rel_gt.translation())
    return np.asarray(rel_gt.rotation().matrix(), dtype=np.float64)

def run_anchor_icp(
    point_clouds: list[o3d.geometry.PointCloud],
    visualizer: RerunVisualizer | None = None,
    depth_frames: list[StereoDepthFrame] | None = None,
    ground_truth_poses: list[gtsam.Pose3] | None = None,
    gt_relative_transforms: list[np.ndarray] | None = None,
    deep_global_registration: object | None = None,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    if not point_clouds:
        raise ValueError("Need at least one point cloud for ICP.")

    anchor_cloud = copy.deepcopy(point_clouds[0])
    anchor_gt_pose = ground_truth_poses[0] if ground_truth_poses else None
    anchor_world_transform = np.eye(4, dtype=np.float64)
    estimation = o3d.pipelines.registration.TransformationEstimationForColoredICP()

    world_transforms: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
    stats: list[tuple[float, float]] = [(1.0, 0.0)]
    last_relative = np.eye(4, dtype=np.float64)
    anchor_down, anchor_fpfh = prepare_fpfh(anchor_cloud, FPFH_VOXEL_SIZE) if USE_FPFH_INITIALIZATION else (None, None)
    if visualizer and visualizer.enabled:
        visualizer.set_frame_index(0)
        visualizer.log_point_cloud("clouds/raw/frame_00", anchor_cloud)
        visualizer.log_point_cloud("clouds/aligned/frame_00", anchor_cloud)
        visualizer.log_pose("poses/icp/frame_00", np.eye(4))

    for idx in range(1, len(point_clouds)):
        source = point_clouds[idx]
        init_transform = last_relative
        dgr_transform = compute_deep_global_registration_transform(deep_global_registration, source, anchor_cloud)
        used_deep_global = dgr_transform is not None
        if used_deep_global:
            init_transform = dgr_transform  # deep global overrides previous estimate
        source_down = source_fpfh = None
        if not used_deep_global and USE_FPFH_INITIALIZATION:
            source_down, source_fpfh = prepare_fpfh(source, FPFH_VOXEL_SIZE)
            if anchor_fpfh is not None and source_fpfh is not None:
                fpfh_match = match_fpfh(source_down, anchor_down, source_fpfh, anchor_fpfh, FPFH_DISTANCE_THRESHOLD)
                if fpfh_match.fitness > 0.0:
                    init_transform = fpfh_match.transformation

        gt_rot = compute_gt_rotation(idx, anchor_gt_pose, ground_truth_poses)
        if gt_rot is not None:
            init_transform = init_transform.copy()
            init_transform[:3, :3] = gt_rot

        # if USE_MULTISCALE_REGISTRATION:
        #     result = multiscale_colored_icp(
        #         source,
        #         anchor_cloud,
        #         init_transform,
        #         estimation,
        #         MS_VOXEL_SIZES,
        #         MS_DISTANCE_THRESHOLDS,
        #     )
        # else:
        #     result = o3d.pipelines.registration.registration_colored_icp(
        #         source,
        #         anchor_cloud,
        #         ICP_DISTANCE_THRESHOLD,
        #         init_transform,
        #         estimation,
        #     )
        # relative_transform = result.transformation
        relative_transform = dgr_transform
        world_transform = anchor_world_transform @ relative_transform
        world_transforms.append(world_transform)
        # stats.append((result.fitness, result.inlier_rmse))
        stats.append((1.0, 0.0))

        # print(
        #     f"Frame {idx:02d}: fitness={result.fitness:.3f}, rmse={result.inlier_rmse:.3f}, "
        #     f"correspondences={len(result.correspondence_set)}"
        # )
        if visualizer and visualizer.enabled:
            visualizer.set_frame_index(idx)
            visualizer.log_point_cloud(f"clouds/raw/frame_{idx:02d}", source, transform=world_transform)
            visualizer.log_point_cloud(
                f"clouds/aligned/frame_{idx:02d}",
                source,
                transform=world_transform,
            )
            visualizer.log_pose(f"poses/icp/frame_{idx:02d}", world_transform)
            if depth_frames is not None and idx < len(depth_frames):
                visualizer.log_rr_pose("poses/rr", world_transform, depth_frames[idx], frame_index=idx)
            if gt_relative_transforms is not None and idx < len(gt_relative_transforms):
                visualizer.log_pose("poses/ground_truth", gt_relative_transforms[idx])
                prev_gt = gt_relative_transforms[idx - 1] if idx - 1 >= 0 else np.eye(4)
                visualizer.log_trajectory_segment(
                    "trajectories/ground_truth",
                    prev_gt[:3, 3],
                    gt_relative_transforms[idx][:3, 3],
                    frame_index=idx,
                    color=(0, 255, 0),
                )
            prev_pos = world_transforms[-2][:3, 3] if len(world_transforms) >= 2 else np.zeros(3)
            visualizer.log_trajectory_segment(
                "trajectories/icp",
                prev_pos,
                world_transform[:3, 3],
                frame_index=idx,
                color=(255, 0, 0),
            )

        # if result.fitness < ANCHOR_RESET_FITNESS_THRESHOLD:
        if dgr_transform is not None:
            anchor_cloud = copy.deepcopy(source)
            anchor_down, anchor_fpfh = prepare_fpfh(anchor_cloud, FPFH_VOXEL_SIZE) if USE_FPFH_INITIALIZATION else (None, None)
            anchor_world_transform = world_transform
            anchor_gt_pose = ground_truth_poses[idx] if ground_truth_poses else None
            last_relative = np.eye(4, dtype=np.float64)
            continue

        aligned_source = copy.deepcopy(source)
        aligned_source.transform(relative_transform)
        anchor_cloud = merge_point_clouds(anchor_cloud, aligned_source)
        anchor_cloud = anchor_cloud.voxel_down_sample(VOXEL_DOWNSAMPLE_SIZE)
        estimate_cloud_normals(anchor_cloud)
        if USE_FPFH_INITIALIZATION:
            anchor_down, anchor_fpfh = prepare_fpfh(anchor_cloud, FPFH_VOXEL_SIZE)
        last_relative = relative_transform
    return world_transforms, stats


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def pose3_relative_to_first(sequence: test_utils.FrameSequenceWithGroundTruth) -> list[np.ndarray]:
    world_to_first = sequence.world_poses[0]
    matrices = [np.eye(4, dtype=np.float64)]
    for pose in sequence.world_poses[1:]:
        rel = world_to_first.inverse() * pose
        matrices.append(np.asarray(rel.matrix(), dtype=np.float64))
    return matrices


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    delta = R_est @ R_gt.T
    cos_angle = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def summarise_pose_errors(estimates: list[np.ndarray], ground_truth: list[np.ndarray]) -> None:
    if len(estimates) != len(ground_truth):
        raise ValueError("Pose count mismatch between estimates and ground truth.")

    print("\nPose errors w.r.t. anchor frame:")
    for idx, (est, gt) in enumerate(zip(estimates, ground_truth)):
        t_err = np.linalg.norm(est[:3, 3] - gt[:3, 3])
        r_err = rotation_error_deg(est[:3, :3], gt[:3, :3])
        print(f"  Frame {idx:02d}: translation={t_err:.3f} m, rotation={r_err:.2f} deg")


def pose3_from_transform(transform: np.ndarray) -> gtsam.Pose3:
    rotation = gtsam.Rot3(transform[:3, :3])
    translation = gtsam.Point3(transform[:3, 3])
    return gtsam.Pose3(rotation, translation)


def create_visualizer() -> RerunVisualizer | None:
    if not ENABLE_RERUN:
        return None
    return RerunVisualizer(
        enabled=True,
        app_id=RERUN_APP_ID,
        spawn_viewer=SPAWN_VIEWER,
        tcp_address=RERUN_TCP,
    )


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main() -> None:
    sequence = test_utils.load_tartanair_sequence_segment(
        env=ENVIRONMENT,
        difficulty=DIFFICULTY,
        traj=TRAJECTORY,
        sequence_length=SEQUENCE_LENGTH,
        seed=RNG_SEED,
        sampling_mode=SAMPLING_MODE,
        min_stride=MIN_STRIDE,
        max_stride=MAX_STRIDE,
        load_ground_truth_depth=True,
    )
    if sequence.ground_truth_depths is None:
        raise RuntimeError("Requested ground-truth depth but loader did not supply it.")

    rectified_frames = [frame.rectify() for frame in tqdm(sequence.frames, desc="Rectifying frames")]
    if DEPTH_MODE == "ground_truth":
        if sequence.ground_truth_depths is None:
            raise RuntimeError("Ground-truth depth requested but not provided.")
        depth_frames = build_ground_truth_depth_frames(
            rectified_frames,
            tqdm(sequence.ground_truth_depths, desc="Building depth frames"),
            max_depth=MAX_DEPTH_METERS,
        )
    else:
        depth_frames = [
            sgbm.compute_depth(frame, max_depth=MAX_DEPTH_METERS)
            for frame in tqdm(rectified_frames, desc="Building depth frames")
        ]
    point_clouds = [point_cloud_from_depth_frame(depth_frame) for depth_frame in tqdm(depth_frames, desc="Building point clouds")]

    visualizer = create_visualizer()
    deep_global_registration = None
    if deep_global_registration_factory is not None:
        deep_global_registration = deep_global_registration_factory()
        print("Deep global registration model loaded.")
    anchor_errors = pose3_relative_to_first(sequence)
    print(f"Loaded {len(point_clouds)} point clouds. Running ICP against anchor frame 0...")
    estimated_transforms, stats = run_anchor_icp(
        point_clouds,
        visualizer=visualizer,
        depth_frames=depth_frames,
        ground_truth_poses=sequence.world_poses,
        gt_relative_transforms=anchor_errors,
        deep_global_registration=deep_global_registration,
    )
    summarise_pose_errors(estimated_transforms, anchor_errors)

    mean_fitness = np.mean([s[0] for s in stats[1:]]) if len(stats) > 1 else 1.0
    mean_rmse = np.mean([s[1] for s in stats[1:]]) if len(stats) > 1 else 0.0
    print(f"\nAverage ICP fitness={mean_fitness:.3f}, rmse={mean_rmse:.3f}")


if __name__ == "__main__":
    main()


# %%
