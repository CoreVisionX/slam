# %%
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import cv2
import gtsam
from gtsam.symbol_shorthand import L, X, V, B
import matplotlib.pyplot as plt
import numpy as np
import torch
import tartanair as ta
import rerun as rr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz import rr_log_pose, rr_log_trajectory

from depth.sgbm import SGBM  # noqa: E402
from registration.registration import (  # noqa: E402
    FeatureFrame,
    FramePair,
    MatchedFramePair,
    RectifiedStereoFrame,
    StereoCalibration,
    StereoDepthFrame,
    StereoFrame
)
from registration.utils import draw_matches, solve_pnp  # noqa: E402
import tests.test_utils as test_utils  # noqa: E402

# TODO: expressing disparity uncertainty for depth measurements in BA properly might help a ton

# %%
# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
NUM_SEQUENCE_SAMPLES = 1
SEQUENCE_LENGTH = 800
ENVIRONMENT = "Hospital"
DIFFICULTY = "diff"
TRAJECTORY = "P1000"
SAMPLING_MODE = "stride"
MIN_STRIDE = 1
MAX_STRIDE = 1
BASE_SEED = 11

USE_IMU_FACTORS = True
DEPTH_MODE = os.environ.get("LOCAL_VO_DEPTH_MODE", "sgbm").strip().lower()
if DEPTH_MODE not in {"sgbm", "ground_truth"}:
    raise ValueError("LOCAL_VO_DEPTH_MODE must be either 'sgbm' or 'ground_truth'.")
USE_GROUND_TRUTH_DEPTH = DEPTH_MODE == "ground_truth"

# Feature detection / tracking
MAX_FEATURE_COUNT = 1024
REFILL_FEATURE_RATIO = 0.8  # refill when remaining tracks fall below 60% of the budget
FEATURE_SUPPRESSION_RADIUS = 8.0
FAST_THRESHOLD = 25
FAST_NONMAX = True
FAST_BORDER = 12

LK_WIN_SIZE = (15, 15)
# LK_WIN_SIZE = (7, 7)
LK_MAX_LEVEL = 5
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)
LK_MIN_EIG_THRESHOLD = 1e-3

MIN_MATCHES_FOR_PNP = 6
MAX_DEPTH = 30.0

# Bundle adjustment
MIN_OBSERVATIONS_PER_LANDMARK = 12
MIN_OBSERVATIONS_PER_FRAME = 5
PROJECTION_NOISE_PX = 1.0
DISPARITY_NOISE_PX = 2.5 # scale based on distance?
PROJECTION_NOISE_HUBER = True
REPROJECTION_GATING_THRESHOLD_PX = 1.0
USE_INLIER_OBSERVATIONS_ONLY = False
POSE_PRIOR_SIGMAS = np.array(
    [
        np.deg2rad(1.0),
        np.deg2rad(1.0),
        np.deg2rad(1.0),
        0.10,
        0.10,
        0.10,
    ],
    dtype=float,
) * 0.001

IMU_GRAVITY_MAGNITUDE = 9.80665
# IMU_ACCEL_NOISE = 0.8e-3
# IMU_GYRO_NOISE = np.deg2rad(0.0006)
# IMU_INTEGRATION_NOISE = 5e-4
# IMU_VELOCITY_PRIOR_SIGMA = 0.1
IMU_BIAS_PRIOR_SIGMAS = np.array(
    [
        1.5e-3,
        1.5e-3,
        1.5e-3,
        1.0e-5,
        1.0e-5,
        1.0e-5,
    ],
    dtype=float,
)
# * 0.001 # imu should be perfect
# IMU_BIAS_PRIOR_SIGMAS = np.array([0.02, 0.02, 0.02, 0.001, 0.001, 0.001], dtype=float)
# IMU_ACCEL_NOISE = 0.04
# IMU_GYRO_NOISE = 1e-3
# IMU_INTEGRATION_NOISE = 5e-0
IMU_ACCEL_NOISE = 2.0e-3
IMU_GYRO_NOISE = 1.7e-4
# IMU_ACCEL_NOISE = 1.0e-3
# IMU_GYRO_NOISE = 8.5e-5
IMU_INTEGRATION_NOISE = 5e-4
IMU_VELOCITY_PRIOR_SIGMA = 0.1
# IMU_BIAS_PRIOR_SIGMAS = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=float) * 100.0

# IMU_GRAVITY_VECTOR = np.array([0.0, 0.0, IMU_GRAVITY_MAGNITUDE], dtype=float)
IMU_GRAVITY_VECTOR = np.array([0.0, IMU_GRAVITY_MAGNITUDE, 0.0], dtype=float)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "klt_local_vo_bundle_adjustment.npz"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Depth source: {'ground_truth' if USE_GROUND_TRUTH_DEPTH else 'sgbm'}")

sgbm = SGBM(num_disparities=16 * 4, block_size=5, image_color="RGB")

_FALSE_ENV_VALUES = {"0", "false", "f", "no"}


def _env_flag(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSE_ENV_VALUES


ENABLE_RERUN = _env_flag("LOCAL_VO_RERUN", False)
RERUN_APP_ID = os.environ.get("LOCAL_VO_RERUN_APP_ID", "local_vo_bundle_adjustment")
RERUN_TCP_ADDRESS = os.environ.get("LOCAL_VO_RERUN_TCP")
RERUN_SPAWN_VIEWER = _env_flag("LOCAL_VO_RERUN_SPAWN", True)
RERUN_ROOT_PATH = os.environ.get("LOCAL_VO_RERUN_PATH", "local_vo_bundle_adjustment")


class LocalVoRerunLogger:
    def __init__(self, *, enabled: bool, app_id: str, spawn_viewer: bool, tcp_address: str | None):
        self._enabled = False
        self._trajectories: dict[str, list[gtsam.Pose3]] = {}
        if not enabled or rr is None or rr_log_pose is None or rr_log_trajectory is None:
            return

        self._enabled = True
        rr.init(app_id, spawn=spawn_viewer)
        if tcp_address:
            rr.connect_grpc(tcp_address)
        rr.log("/", rr.ViewCoordinates.RDF)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log_pose_with_trajectory(
        self,
        path: str,
        frame_index: int,
        pose: gtsam.Pose3,
        frame: RectifiedStereoFrame | StereoDepthFrame,
    ) -> None:
        if not self._enabled:
            return
        assert rr is not None
        rr.set_time("frame", sequence=frame_index)
        rr_log_pose(path, pose, frame)
        trajectory = self._trajectories.setdefault(path, [])
        trajectory.append(pose)
        rr_log_trajectory(f"{path}/trajectory", trajectory)

    def log_pnp_sequence(
        self,
        path: str,
        depth_frames: list[StereoDepthFrame],
        sequence_results: list[dict[str, Any]],
    ) -> None:
        if not self._enabled or not depth_frames:
            return

        self.log_pose_with_trajectory(path, 0, gtsam.Pose3.Identity(), depth_frames[0])
        for result in sequence_results:
            if result.get("status") != "success":
                continue
            frame_index = result.get("frame_index")
            if not isinstance(frame_index, int) or frame_index >= len(depth_frames) or frame_index < 0:
                continue
            estimated_pose = result.get("estimated_pose")
            if estimated_pose is None:
                continue
            self.log_pose_with_trajectory(path, frame_index, estimated_pose, depth_frames[frame_index])

    def log_bundle_adjustment(
        self,
        path: str,
        depth_frames: list[StereoDepthFrame],
        ba_result: dict[str, Any],
    ) -> None:
        if not self._enabled or not depth_frames:
            return

        for stage_label, pose_dict_key in (("initial", "initial_pose_dict"), ("optimized", "optimized_pose_dict")):
            pose_dict = ba_result.get(pose_dict_key)
            if not pose_dict:
                continue
            stage_path = f"{path}/{stage_label}"
            for frame_index in sorted(pose_dict):
                if frame_index < 0 or frame_index >= len(depth_frames):
                    continue
                pose = pose_dict[frame_index]
                self.log_pose_with_trajectory(stage_path, frame_index, pose, depth_frames[frame_index])


def create_local_vo_rerun_logger() -> LocalVoRerunLogger | None:
    if not ENABLE_RERUN:
        return None
    return LocalVoRerunLogger(
        enabled=True,
        app_id=RERUN_APP_ID,
        spawn_viewer=RERUN_SPAWN_VIEWER,
        tcp_address=RERUN_TCP_ADDRESS,
    )


rerun_logger = create_local_vo_rerun_logger()


def _log_sequence_results_to_rerun(
    logger: LocalVoRerunLogger,
    variant_label: str,
    depth_frames: list[StereoDepthFrame],
    sequence_results: list[dict[str, Any]],
) -> None:
    if not logger.enabled:
        return
    base_path = f"{RERUN_ROOT_PATH}/{variant_label}/pnp"
    logger.log_pnp_sequence(base_path, depth_frames, sequence_results)


def _log_bundle_adjustment_to_rerun(
    logger: LocalVoRerunLogger,
    variant_label: str,
    depth_frames: list[StereoDepthFrame],
    ba_result: dict[str, Any],
    stage_suffix: str,
) -> None:
    if not logger.enabled:
        return
    base_path = f"{RERUN_ROOT_PATH}/{variant_label}/ba/{stage_suffix}"
    logger.log_bundle_adjustment(base_path, depth_frames, ba_result)



# %%
# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
@dataclass
class TrackObservation:
    keypoint: np.ndarray
    depth: float


@dataclass
class FeatureTrack:
    track_id: int
    anchor_frame: int
    anchor_keypoint: np.ndarray
    anchor_depth: float
    anchor_point3: np.ndarray
    anchor_color: np.ndarray
    observations: dict[int, TrackObservation] = field(default_factory=dict)
    active: bool = True

    def add_observation(self, frame_idx: int, point: np.ndarray, depth: float) -> None:
        self.observations[frame_idx] = TrackObservation(
            keypoint=np.asarray(point, dtype=np.float32),
            depth=float(depth),
        )

def detect_fast_keypoints(
    gray_image: np.ndarray,
    max_features: int,
    threshold: int,
    nonmax: bool,
    border: int,
) -> np.ndarray:
    detector = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax)
    keypoints = detector.detect(gray_image)
    if not keypoints:
        raise RuntimeError("FAST detector did not return any keypoints.")

    points = cv2.KeyPoint_convert(keypoints)
    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)

    if border > 0:
        h, w = gray_image.shape
        inside = (
            (points[:, 0] >= border)
            & (points[:, 0] < w - border)
            & (points[:, 1] >= border)
            & (points[:, 1] < h - border)
        )
        points = points[inside]
        responses = responses[inside]

    if points.size == 0:
        raise RuntimeError("All FAST keypoints were rejected by the border filter.")

    if points.shape[0] > max_features:
        order = np.argsort(responses)[::-1][:max_features]
        points = points[order]

    return points.astype(np.float32)


def extract_keypoint_attributes(
    depth_frame: StereoDepthFrame,
    keypoints: np.ndarray,
    max_depth: float,
) -> dict[str, Any]:
    h, w = depth_frame.left_depth.shape
    clipped = np.clip(np.round(keypoints), [0, 0], [w - 1, h - 1]).astype(int)
    rows = clipped[:, 1]
    cols = clipped[:, 0]

    depths = depth_frame.left_depth[rows, cols]
    xyz = depth_frame.left_depth_xyz[rows, cols]
    colors = depth_frame.left_rect[rows, cols]

    valid = np.isfinite(depths) & (depths > 0.0) & (depths <= max_depth)
    filtered = {
        "keypoints": keypoints[valid],
        "keypoints_depth": depths[valid],
        "keypoints_3d": xyz[valid],
        "keypoints_color": colors[valid],
        "valid_mask": valid,
    }
    return filtered


def build_feature_frame(
    depth_frame: StereoDepthFrame,
    attributes: dict[str, Any],
) -> FeatureFrame:
    features: dict[str, Any] = {
        "keypoints": torch.from_numpy(attributes["keypoints"]).float(),
        "keypoints_depth": attributes["keypoints_depth"],
        "keypoints_3d": attributes["keypoints_3d"],
        "keypoints_color": attributes["keypoints_color"],
        "image_size": depth_frame.left_rect.shape[:2],
    }
    return FeatureFrame(
        left=None,
        right=None,
        left_rect=None,
        right_rect=None,
        left_depth=None,
        left_depth_xyz=None,
        calibration=depth_frame.calibration,
        features=features,
    )


def remap_ground_truth_depth(raw_depth: np.ndarray, calibration: StereoCalibration) -> np.ndarray:
    depth = np.asarray(raw_depth, dtype=np.float32)
    expected_shape = (calibration.height, calibration.width)
    if depth.shape != expected_shape:
        raise ValueError(
            f"Depth map shape {depth.shape} does not match calibration size {expected_shape}."
        )
    remapped = cv2.remap(
        depth,
        calibration.map_left_x,
        calibration.map_left_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return remapped


def depth_map_to_xyz(depth_map: np.ndarray, calibration_matrix: np.ndarray) -> np.ndarray:
    h, w = depth_map.shape
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)
    fx = float(calibration_matrix[0, 0])
    fy = float(calibration_matrix[1, 1])
    cx = float(calibration_matrix[0, 2])
    cy = float(calibration_matrix[1, 2])
    if np.isclose(fx, 0.0) or np.isclose(fy, 0.0):
        raise ValueError("Invalid rectified intrinsics; fx/fy must be non-zero.")
    z = depth_map
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    xyz = np.stack((x, y, z), axis=-1).astype(np.float32, copy=False)
    invalid = ~np.isfinite(z)
    xyz[invalid] = np.nan
    return xyz


def build_ground_truth_depth_frame(
    rectified_frame: RectifiedStereoFrame,
    raw_depth_map: np.ndarray,
    max_depth: float,
) -> StereoDepthFrame:
    depth_rect = remap_ground_truth_depth(raw_depth_map, rectified_frame.calibration)
    depth_rect = depth_rect.astype(np.float32, copy=False)
    invalid = (~np.isfinite(depth_rect)) | (depth_rect <= 0.0) | (depth_rect > max_depth)
    depth_rect[invalid] = np.nan
    # depth_xyz = depth_map_to_xyz(depth_rect, rectified_frame.calibration.K_left_rect)
    depth_xyz = depth_map_to_xyz(depth_rect, rectified_frame.calibration.K_left)
    depth_xyz[invalid, :] = np.nan
    return StereoDepthFrame(
        left=rectified_frame.left,
        right=rectified_frame.right,
        calibration=rectified_frame.calibration,
        left_rect=rectified_frame.left_rect,
        right_rect=rectified_frame.right_rect,
        left_depth=depth_rect,
        left_depth_xyz=depth_xyz,
    )


def build_ground_truth_depth_frames(
    rectified_frames: list[RectifiedStereoFrame],
    raw_depth_maps: list[np.ndarray],
    max_depth: float,
) -> list[StereoDepthFrame]:
    if len(rectified_frames) != len(raw_depth_maps):
        raise ValueError("Depth map count does not match the number of rectified frames.")
    return [
        build_ground_truth_depth_frame(rectified_frame, raw_depth, max_depth)
        for rectified_frame, raw_depth in zip(rectified_frames, raw_depth_maps)
    ]


def create_preintegration_params() -> gtsam.PreintegrationParams:
    params = gtsam.PreintegrationCombinedParams.MakeSharedU(IMU_GRAVITY_MAGNITUDE)
    params.n_gravity = IMU_GRAVITY_VECTOR
    params.setAccelerometerCovariance(np.eye(3) * (IMU_ACCEL_NOISE ** 2))
    params.setGyroscopeCovariance(np.eye(3) * (IMU_GYRO_NOISE ** 2))
    params.setIntegrationCovariance(np.eye(3) * (IMU_INTEGRATION_NOISE ** 2))
    return params


def _integrate_imu_batch(
    pim: gtsam.PreintegratedImuMeasurements,
    batch: test_utils.FrameImuMeasurements,
) -> None:
    if batch is None:
        raise ValueError("IMU batch missing while attempting to integrate measurements.")

    # current_time = batch.frame_timestamp
    # if batch.timestamps.size == 0:
        # dt = float(next_timestamp - current_time)
        # if dt > 0.0:
        #     pim.integrateMeasurement(np.zeros(3), np.zeros(3), dt)
        # return

    # integrated = 0
    # for sample_time, accel, gyro in zip(
    #     batch.timestamps,
    #     batch.linear_accelerations,
    #     batch.angular_velocities,
    # ):
    #     dt = float(sample_time - current_time)
    #     if dt > 0.0:
    #         pim.integrateMeasurement(accel, gyro, dt)
    #     current_time = float(sample_time)
    #     integrated += 1

    integrated = 0
    # print(f'{len(batch.timestamps)} IMU samples')
    for i in range(len(batch)):
        pim.integrateMeasurement(
            batch.linear_accelerations[i],
            batch.angular_velocities[i],
            deltaT=batch.dts[i],
        )
        integrated += 1
    assert integrated == 10

# def preintegrate_between_frames(
#     sequence: test_utils.FrameSequenceWithGroundTruth[StereoFrame],  # type: ignore[type-arg]
#     start_idx: int,
#     end_idx: int,
#     params: gtsam.PreintegrationParams,
#     bias: gtsam.imuBias.ConstantBias,
# ) -> gtsam.PreintegratedImuMeasurements:
#     if sequence.full_imu_measurements is None or sequence.frame_timestamps is None:
#         raise ValueError("IMU measurements unavailable for this sequence.")
#     if end_idx <= start_idx:
#         raise ValueError("end_idx must be greater than start_idx for IMU preintegration.")

#     pim = gtsam.PreintegratedImuMeasurements(params, bias)
#     imu_batches = sequence.full_imu_measurements
#     timestamps = sequence.frame_timestamps
#     frame_indices = sequence.frame_indices

#     assert end_idx == start_idx + 1

#     batch = imu_batches[frame_indices[end_idx]]
#     _integrate_imu_batch(pim, batch)

#     # raw_start = frame_indices[start_idx + 1]
#     # raw_end = frame_indices[end_idx + 1]

#     # for raw_idx in range(raw_start, raw_end):
#     #     if raw_idx >= len(imu_batches):
#     #         break
#     #     batch = imu_batches[raw_idx]
#     #     if raw_idx + 1 >= len(timestamps):
#     #         break
#     #     _integrate_imu_batch(pim, batch)

#     return pim


def preintegrate_between_frames(sequence, start_idx, end_idx, params, bias, time_offset: float = 0.0):
    if end_idx <= start_idx:
        raise ValueError("end_idx must be greater than start_idx")

    pim = gtsam.PreintegratedImuMeasurements(params, bias)

    t_start = float(sequence.imu_measurements[start_idx].frame_timestamp + time_offset)
    t_end   = float(sequence.imu_measurements[end_idx].frame_timestamp   + time_offset)
    assert t_end > t_start

    # We’ll walk the raw IMU stream across batches and integrate only the slice inside [t_start, t_end]
    imu_batches = sequence.full_imu_measurements  # or the stream you use now
    # Find raw span [raw_lo, raw_hi) that covers t_start..t_end
    raw_lo = int(sequence.frame_indices[start_idx])
    raw_hi = int(sequence.frame_indices[end_idx])

    # Helper to clamp dt within the window and use sample at index k-1
    def integrate_window(ts, acc, gyro, win_lo, win_hi):
        # ts: monotonic np.array of times (float), same length as acc/gyro
        # We integrate from max(win_lo, ts[k-1]) to min(win_hi, ts[k]) using (acc[k-1], gyro[k-1])
        if ts.size < 2:
            return
        # Ensure we start with a time just before win_lo
        t_prev = max(win_lo, float(ts[0]))
        for k in range(1, len(ts)):
            t_curr = float(ts[k])
            if t_curr <= win_lo:
                t_prev = win_lo
                continue
            if t_prev >= win_hi:
                break
            dt = min(t_curr, win_hi) - max(t_prev, win_lo)
            if dt > 0.0 and np.isfinite(dt):
                pim.integrateMeasurement(acc[k-1], gyro[k-1], float(dt))
            t_prev = t_curr

    for raw_idx in range(raw_lo, raw_hi + 1):
        batch = imu_batches[raw_idx]
        ts   = batch.timestamps.astype(np.float64)
        acc  = batch.linear_accelerations
        gyro = batch.angular_velocities
        if ts.size == 0:
            continue
        # Fast reject if this batch is completely outside the window
        if ts[-1] <= t_start or ts[0] >= t_end:
            continue
        integrate_window(ts, acc, gyro, t_start, t_end)

    # Sanity: preintegrated Δt should match (t_end - t_start)
    dt_frames = t_end - t_start
    if np.isfinite(dt_frames):
        # let ~1e-4s slack
        assert abs(pim.deltaTij() - dt_frames) < 1e-4, f"IMU Δt {pim.deltaTij():.6f} vs expected {dt_frames:.6f}"

    return pim


def estimate_sequence_velocities(
    sequence: test_utils.FrameSequenceWithGroundTruth[StereoFrame],  # type: ignore[type-arg]
) -> np.ndarray:
    if sequence.full_imu_measurements is not None:
        velocities = []
        for raw_idx in sequence.frame_indices:
            measurement = sequence.full_imu_measurements[raw_idx]
            velocities.append(np.asarray(measurement.world_velocity, dtype=np.float64))
        return np.asarray(velocities, dtype=np.float64)

    positions = np.array(
        [to_numpy_vec3(pose.translation()) for pose in sequence.world_poses],
        dtype=np.float64,
    )
    timestamps = np.array(
        [batch.frame_timestamp for batch in sequence.imu_measurements]
        if sequence.imu_measurements
        else np.arange(sequence.length, dtype=np.float64),
        dtype=np.float64,
    )

    velocities = np.zeros_like(positions)
    for idx in range(1, sequence.length):
        dt = float(timestamps[idx] - timestamps[idx - 1])
        assert dt > 0.0 and np.isfinite(dt), f"Invalid time delta: {dt} at index {idx}"
        velocities[idx] = (positions[idx] - positions[idx - 1]) / dt
    if sequence.length > 1:
        velocities[0] = velocities[1]
    return velocities


def integrate_inertial_trajectory(
    sequence: test_utils.FrameSequenceWithGroundTruth[StereoFrame],  # type: ignore[type-arg]
) -> list[gtsam.Pose3] | None:
    gt_world_to_first = sequence.world_poses[0]

    params = create_preintegration_params()
    bias = gtsam.imuBias.ConstantBias()
    velocities = estimate_sequence_velocities(sequence)

    # inertial_poses: list[gtsam.Pose3] = []

    # # preintegrate relative to the previous ground truth pose
    # for idx in range(1, sequence.length):
    #     gt_first_to_prev = gt_world_to_first.inverse() * sequence.world_poses[idx - 1]
    #     initial_velocity = R_world_to_first @ velocities[idx - 1]

    #     nav_state = gtsam.NavState(gt_first_to_prev, initial_velocity)

    #     pim = preintegrate_between_frames(sequence, idx - 1, idx, params, bias)
    #     nav_state = pim.predict(nav_state, bias)
    #     inertial_poses.append(nav_state.pose())

    nav_states = [gtsam.NavState(gtsam.Pose3(gt_world_to_first.rotation(), np.zeros(3)), sequence.imu_measurements[0].world_velocity)]
    translation_gt_world_to_first = gtsam.Pose3(gtsam.Rot3.Identity(), gt_world_to_first.translation()).inverse()
    # next_nav_states = [
        # gtsam.NavState(translation_gt_world_to_first * sequence.world_poses[i], sequence.imu_measurements[i].world_velocity) for i in range(1, sequence.length)
    # ]
    # nav_states += next_nav_states

    # imu_params = create_preintegration_params()
    # bias = gtsam.imuBias.ConstantBias()

    # # preintegrate relative to the previous preintegrated pose
    for idx in range(1, sequence.length):
        batch = sequence.imu_measurements[idx]

        # nav_state = gtsam.NavState(translation_gt_world_to_first * sequence.world_poses[idx - 1], sequence.imu_measurements[idx - 1].world_velocity)
        nav_state = nav_states[-1]

        # pose = nav_state.pose()
        # pose_trans = pose.translation() + sequence.imu_measurements[idx - 1].world_velocity * 0.1
        # nav_state = gtsam.NavState(gtsam.Pose3(pose.rotation(), pose_trans), sequence.imu_measurements[idx - 1].world_velocity)
        # nav_states.append(nav_state)

        pim = gtsam.PreintegratedImuMeasurements(params, bias)
        _integrate_imu_batch(pim, batch)
        # pim = preintegrate_between_frames(sequence, idx - 1, idx, params, bias)
        # pim.integrateMeasurement(np.zeros(3), np.zeros(3), 0.1)
        nav_states.append(pim.predict(nav_state, bias))

    inertial_poses = [nav_state.pose() for nav_state in nav_states]

    return inertial_poses


def _filter_keypoints_by_distance(
    candidate_points: np.ndarray,
    existing_points: np.ndarray,
    min_radius: float,
) -> np.ndarray:
    if candidate_points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if existing_points.size == 0:
        return candidate_points.astype(np.float32)

    kept: list[np.ndarray] = []
    current = existing_points.reshape(-1, 2).astype(np.float32)
    min_radius_sq = float(min_radius * min_radius)
    for point in candidate_points.astype(np.float32):
        if current.size:
            deltas = current - point
            if np.min(np.sum(deltas * deltas, axis=1)) < min_radius_sq:
                continue
        kept.append(point)
        if current.size:
            current = np.vstack([current, point.reshape(1, 2)])
        else:
            current = point.reshape(1, 2)
    if not kept:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(kept, dtype=np.float32)


def _sample_depth_at_point(depth_frame: StereoDepthFrame, point: np.ndarray) -> float:
    depth_map = depth_frame.left_depth
    if depth_map.size == 0:
        return float("nan")
    h, w = depth_map.shape
    u = float(point[0])
    v = float(point[1])
    if not (0 <= u < w and 0 <= v < h):
        return float("nan")
    col = int(np.clip(np.round(u), 0, w - 1))
    row = int(np.clip(np.round(v), 0, h - 1))
    depth_value = float(depth_map[row, col])
    if not np.isfinite(depth_value) or depth_value <= 0.0:
        return float("nan")
    return depth_value


def track_features_with_refill(
    rectified_frames: list[RectifiedStereoFrame],
    depth_frames: list[StereoDepthFrame],
) -> tuple[list[dict[int, TrackObservation]], dict[int, FeatureTrack]]:
    if len(rectified_frames) != len(depth_frames):
        raise ValueError("Number of rectified frames and depth frames must match.")

    gray_sequence = [cv2.cvtColor(frame.left_rect, cv2.COLOR_RGB2GRAY) for frame in rectified_frames]
    frame_observations: list[dict[int, TrackObservation]] = []
    tracks: dict[int, FeatureTrack] = {}
    next_track_id = 0

    prev_gray: np.ndarray | None = None
    prev_points = np.empty((0, 1, 2), dtype=np.float32)
    prev_ids = np.empty(0, dtype=int)

    lk_params = dict(
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA,
        minEigThreshold=LK_MIN_EIG_THRESHOLD,
    )
    refill_threshold = int(np.floor(REFILL_FEATURE_RATIO * MAX_FEATURE_COUNT))

    for frame_idx, (gray, depth_frame) in enumerate(zip(gray_sequence, depth_frames)):
        frame_obs: dict[int, TrackObservation] = {}
        tracked_points: list[np.ndarray] = []
        tracked_ids: list[int] = []

        if prev_gray is not None and prev_points.size and prev_ids.size:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
            if next_points is None or status is None:
                status_mask = np.zeros(prev_ids.shape[0], dtype=bool)
                next_points_arr = np.zeros((prev_ids.shape[0], 2), dtype=np.float32)
            else:
                status_mask = status.reshape(-1).astype(bool)
                next_points_arr = next_points.reshape(-1, 2).astype(np.float32)

            for idx, track_id in enumerate(prev_ids):
                if idx >= status_mask.size or not status_mask[idx]:
                    tracks[track_id].active = False
                    continue
                point = next_points_arr[idx]
                depth_value = _sample_depth_at_point(depth_frame, point)
                tracked_points.append(point)
                tracked_ids.append(track_id)
                tracks[track_id].add_observation(frame_idx, point, depth_value)
                frame_obs[track_id] = tracks[track_id].observations[frame_idx]
        elif frame_idx != 0:
            # No previous tracks to propagate.
            prev_ids = np.empty(0, dtype=int)
            prev_points = np.empty((0, 1, 2), dtype=np.float32)

        existing_points = np.asarray(tracked_points, dtype=np.float32) if tracked_points else np.empty((0, 2), dtype=np.float32)

        def add_new_tracks(budget: int) -> tuple[list[int], np.ndarray]:
            nonlocal next_track_id, existing_points
            if budget <= 0:
                return [], np.empty((0, 2), dtype=np.float32)
            detection_quota = max(MAX_FEATURE_COUNT, budget * 2)
            try:
                candidate_points = detect_fast_keypoints(
                    gray,
                    max_features=detection_quota,
                    threshold=FAST_THRESHOLD,
                    nonmax=FAST_NONMAX,
                    border=FAST_BORDER,
                )
            except RuntimeError:
                if frame_idx == 0:
                    raise
                return [], np.empty((0, 2), dtype=np.float32)

            filtered_candidates = _filter_keypoints_by_distance(
                candidate_points,
                existing_points,
                FEATURE_SUPPRESSION_RADIUS,
            )
            if filtered_candidates.size == 0:
                return [], np.empty((0, 2), dtype=np.float32)

            attributes = extract_keypoint_attributes(depth_frame, filtered_candidates, max_depth=MAX_DEPTH)
            new_keypoints = attributes["keypoints"]
            if new_keypoints.size == 0:
                return [], np.empty((0, 2), dtype=np.float32)

            depths = attributes["keypoints_depth"]
            xyz = attributes["keypoints_3d"]
            colors = attributes["keypoints_color"]

            take = min(budget, new_keypoints.shape[0])
            new_ids: list[int] = []
            new_points_list: list[np.ndarray] = []
            for idx_new in range(take):
                keypoint = new_keypoints[idx_new]
                depth_value = float(depths[idx_new])
                point3 = np.asarray(xyz[idx_new], dtype=np.float64)
                color = np.asarray(colors[idx_new])

                track = FeatureTrack(
                    track_id=next_track_id,
                    anchor_frame=frame_idx,
                    anchor_keypoint=keypoint.astype(np.float32),
                    anchor_depth=depth_value,
                    anchor_point3=point3,
                    anchor_color=color,
                )
                track.add_observation(frame_idx, keypoint, depth_value)
                tracks[next_track_id] = track
                new_ids.append(next_track_id)
                new_points_list.append(keypoint.astype(np.float32))
                frame_obs[next_track_id] = track.observations[frame_idx]
                next_track_id += 1

            if new_points_list:
                new_points_arr = np.asarray(new_points_list, dtype=np.float32)
                existing_points = (
                    np.vstack([existing_points, new_points_arr]) if existing_points.size else new_points_arr
                )
            else:
                new_points_arr = np.empty((0, 2), dtype=np.float32)
            return new_ids, new_points_arr

        need_refill = len(tracked_ids) <= refill_threshold
        budget = MAX_FEATURE_COUNT - len(tracked_ids) if need_refill else 0
        new_ids, new_points = add_new_tracks(budget)
        if new_ids:
            tracked_ids.extend(new_ids)
            if new_points.size:
                tracked_points.extend(new_points.reshape(-1, 2))

        active_ids = np.asarray(tracked_ids, dtype=int)
        if tracked_points:
            stacked_points = np.asarray(tracked_points, dtype=np.float32).reshape(-1, 1, 2)
        else:
            stacked_points = np.empty((0, 1, 2), dtype=np.float32)

        for track_id in active_ids:
            tracks[track_id].active = True

        frame_observations.append(frame_obs)
        prev_gray = gray
        prev_points = stacked_points
        prev_ids = active_ids

    return frame_observations, tracks


def make_feature_frame_for_view(
    rectified_frame: RectifiedStereoFrame,
    keypoints: np.ndarray,
) -> FeatureFrame:
    features: dict[str, Any] = {
        "keypoints": torch.from_numpy(keypoints.astype(np.float32)),
        "image_size": rectified_frame.left_rect.shape[:2],
    }
    return FeatureFrame(
        left=None,
        right=None,
        left_rect=None,
        right_rect=None,
        left_depth=None,
        left_depth_xyz=None,
        calibration=rectified_frame.calibration,
        features=features,
    )


def to_numpy_vec3(vec: Any) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.size == 3:
        return arr
    if hasattr(vec, "x") and hasattr(vec, "y") and hasattr(vec, "z"):
        return np.array([vec.x(), vec.y(), vec.z()], dtype=np.float64)
    raise TypeError(f"Cannot convert value of type {type(vec)} to a 3-vector.")


def imu_bias_to_numpy(bias: gtsam.imuBias.ConstantBias) -> np.ndarray:
    accel = to_numpy_vec3(bias.accelerometer())
    gyro = to_numpy_vec3(bias.gyroscope())
    return np.concatenate([accel, gyro], axis=0)


def clone_pose(pose: gtsam.Pose3) -> gtsam.Pose3:
    rotation = gtsam.Rot3(pose.rotation().matrix())
    translation = to_numpy_vec3(pose.translation())
    return gtsam.Pose3(rotation, translation)


def pose_error_components(pose: gtsam.Pose3) -> tuple[np.ndarray, np.ndarray]:
    translation = to_numpy_vec3(pose.translation())
    rotation = gtsam.Rot3.Logmap(pose.rotation())
    return translation, rotation


def summarise_tracking_results(
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
    track_history: list[dict[int, TrackObservation]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    sequence_length = sequence_sample.length
    translation_errors = np.full((sequence_length, 3), np.nan, dtype=np.float64)
    rotation_errors = np.full((sequence_length, 3), np.nan, dtype=np.float64)
    translation_norms = np.full(sequence_length, np.nan, dtype=np.float64)
    rotation_norms_rad = np.full(sequence_length, np.nan, dtype=np.float64)
    inlier_counts = np.zeros(sequence_length, dtype=int)
    active_track_counts = np.array([len(observations) for observations in track_history], dtype=int)

    for result in results:
        idx = result["frame_index"]
        if result["status"] != "success":
            continue
        err = result["pose_error"]
        trans, rot = pose_error_components(err)
        translation_errors[idx] = trans
        rotation_errors[idx] = rot
        translation_norms[idx] = np.linalg.norm(trans)
        rotation_norms_rad[idx] = np.linalg.norm(rot)
        inlier_counts[idx] = result["matches_after_filter"]

    summary = {
        "translation_errors": translation_errors,
        "rotation_errors": rotation_errors,
        "translation_norms": translation_norms,
        "rotation_norms_rad": rotation_norms_rad,
        "rotation_norms_deg": np.rad2deg(rotation_norms_rad),
        "inlier_counts": inlier_counts,
        "active_track_counts": active_track_counts,
    }

    print("------------------------------------------------------------")
    print("Per-frame KLT track counts:", active_track_counts.tolist())
    successful = [res for res in results if res["status"] == "success"]
    print(f"Successful PnP frames: {len(successful)} / {sequence_length - 1}")
    valid_trans = translation_norms[~np.isnan(translation_norms)]
    valid_rot = rotation_norms_rad[~np.isnan(rotation_norms_rad)]
    if valid_trans.size:
        print(f"Average translation error: {np.mean(valid_trans):.3f} m")
    if valid_rot.size:
        print(f"Average rotation error: {np.rad2deg(np.mean(valid_rot)):.2f} deg")
    print("------------------------------------------------------------")

    return summary


def estimate_sequence_poses(
    rectified_frames: list[RectifiedStereoFrame],
    depth_frames: list[StereoDepthFrame],
    track_history: list[dict[int, TrackObservation]],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    first_gt_pose = gtsam.Pose3(sequence_sample.world_poses[0].rotation(), np.zeros(3))
    world_pose = gtsam.Pose3(sequence_sample.world_poses[0].rotation(), np.zeros(3))

    for frame_idx in range(1, len(rectified_frames)):
        prev_frame_idx = frame_idx - 1
        prev_world_pose = world_pose
        prev_obs = track_history[frame_idx - 1]
        curr_obs = track_history[frame_idx]
        common_track_ids = sorted(set(prev_obs.keys()) & set(curr_obs.keys()))
        if not common_track_ids:
            results.append(
                {
                    "frame_index": frame_idx,
                    "frame_id": sequence_sample.frame_ids[frame_idx],
                    "status": "no_tracks",
                    "active_track_count": 0,
                }
            )
            continue

        prev_points = np.asarray(
            [prev_obs[track_id].keypoint for track_id in common_track_ids],
            dtype=np.float32,
        )
        curr_points = np.asarray(
            [curr_obs[track_id].keypoint for track_id in common_track_ids],
            dtype=np.float32,
        )

        attributes = extract_keypoint_attributes(depth_frames[frame_idx - 1], prev_points, max_depth=MAX_DEPTH)
        valid_mask = attributes["valid_mask"]
        if not np.any(valid_mask):
            results.append(
                {
                    "frame_index": frame_idx,
                    "frame_id": sequence_sample.frame_ids[frame_idx],
                    "status": "no_valid_depth",
                    "active_track_count": len(common_track_ids),
                }
            )
            continue

        prev_valid = attributes["keypoints"]
        curr_valid = curr_points[valid_mask]
        valid_track_ids = np.asarray(common_track_ids, dtype=int)[valid_mask]
        if prev_valid.shape[0] < MIN_MATCHES_FOR_PNP:
            results.append(
                {
                    "frame_index": frame_idx,
                    "frame_id": sequence_sample.frame_ids[frame_idx],
                    "status": "insufficient_tracks",
                    "active_track_count": prev_valid.shape[0],
                }
            )
            continue

        first_frame = build_feature_frame(depth_frames[prev_frame_idx], attributes)
        second_frame = make_feature_frame_for_view(rectified_frames[frame_idx], curr_valid)
        match_indices = np.arange(prev_valid.shape[0], dtype=int)
        matches = np.stack([match_indices, match_indices], axis=1)
        matched_pair = MatchedFramePair(
            first=first_frame,
            second=second_frame,
            matches=matches,
        )

        try:
            relative_pose, inlier_pair = solve_pnp(matched_pair)
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "frame_index": frame_idx,
                    "frame_id": sequence_sample.frame_ids[frame_idx],
                    "status": "pnp_failed",
                    "active_track_count": prev_valid.shape[0],
                    "error": str(exc),
                }
            )
            continue

        estimated_pose = prev_world_pose.compose(relative_pose)
        world_pose = estimated_pose
        # ground_truth_pose = sequence_sample.relative_pose(0, frame_idx)
        ground_truth_pose = first_gt_pose.inverse() * sequence_sample.world_poses[frame_idx]
        pose_error = ground_truth_pose.between(estimated_pose)

        results.append(
            {
                "frame_index": frame_idx,
                "frame_id": sequence_sample.frame_ids[frame_idx],
                "status": "success",
                "active_track_count": prev_valid.shape[0],
                "matches_before_filter": len(common_track_ids),
                "matches_after_filter": inlier_pair.matches.shape[0],
                "estimated_pose": estimated_pose,
                "ground_truth_pose": ground_truth_pose,
                "pose_error": pose_error,
                "matched_pair": inlier_pair,
                "track_ids": valid_track_ids,
                "relative_pose": relative_pose,
                "anchor_frame_index": prev_frame_idx,
            }
        )

    return results


def compute_pose_initializations(
    rotate_world_to_first: gtsam.Pose3,
    sequence_results: list[dict[str, Any]],
    sequence_length: int,
) -> list[gtsam.Pose3]:
    initializations: list[gtsam.Pose3] = [rotate_world_to_first]
    result_by_frame = {result["frame_index"]: result for result in sequence_results if result["status"] == "success"}
    current_pose = rotate_world_to_first

    for frame_idx in range(1, sequence_length):
        if frame_idx in result_by_frame:
            current_pose = result_by_frame[frame_idx]["estimated_pose"]
        initializations.append(current_pose)

    return initializations


def select_frames_for_ba(
    track_history: list[dict[int, TrackObservation]],
) -> list[int]:
    selected = []
    for idx, observations in enumerate(track_history):
        visible = len(observations)
        if visible >= MIN_OBSERVATIONS_PER_FRAME or idx == 0:
            selected.append(idx)
    if 0 not in selected:
        selected.insert(0, 0)
    return selected


def build_calibration(frame: RectifiedStereoFrame) -> gtsam.Cal3_S2:
    K = frame.calibration.K_left_rect
    # K = frame.calibration.K_left
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    skew = float(K[0, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return gtsam.Cal3_S2(fx, fy, skew, cx, cy)


def build_stereo_calibration(frame: RectifiedStereoFrame) -> gtsam.Cal3_S2Stereo:
    calib = frame.calibration
    K = calib.K_left_rect
    # K = calib.K_left
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    skew = float(K[0, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    baseline = float(np.abs(calib.P_right[0, 3] / calib.P_right[0, 0]))
    if baseline <= 0.0:
        baseline = float(np.linalg.norm(calib.T.reshape(-1)))
    return gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, baseline)


def point3_like_to_numpy(point: Any) -> np.ndarray:
    return to_numpy_vec3(point)


def compute_reprojection_errors(
    pose_dict: dict[int, gtsam.Pose3],
    landmark_positions: list[np.ndarray],
    observations: list[tuple[int, int, np.ndarray]],
    calibration: gtsam.Cal3_S2,
) -> np.ndarray:
    errors: list[float] = []
    for frame_idx, landmark_idx, measurement in observations:
        if frame_idx not in pose_dict:
            continue
        pose = pose_dict[frame_idx]
        point = point3_like_to_numpy(landmark_positions[landmark_idx])
        camera = gtsam.PinholeCameraCal3_S2(pose, calibration)
        try:
            projected = camera.project(point)
        except RuntimeError:
            continue
        if hasattr(projected, "vector"):
            proj_vec = projected.vector()
        else:
            proj_vec = np.asarray(projected, dtype=np.float64)
        residual = proj_vec - measurement
        errors.append(float(np.linalg.norm(residual)))
    return np.asarray(errors, dtype=np.float64)


def run_bundle_adjustment(
    rectified_frames: list[RectifiedStereoFrame],
    track_history: list[dict[int, TrackObservation]],
    tracks: dict[int, FeatureTrack],
    sequence_results: list[dict[str, Any]],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
    use_imu: bool = True,
    log_prefix: str | None = None,
) -> dict[str, Any]:
    frames_for_ba = select_frames_for_ba(track_history)
    calibration = build_calibration(rectified_frames[0])
    stereo_calibration = build_stereo_calibration(rectified_frames[0])

    if PROJECTION_NOISE_HUBER:
        measurement_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber(1.345),
            gtsam.noiseModel.Isotropic.Sigma(2, PROJECTION_NOISE_PX),
        )
        stereo_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber(1.345),
            gtsam.noiseModel.Diagonal.Sigmas(
                np.array([PROJECTION_NOISE_PX, PROJECTION_NOISE_PX, DISPARITY_NOISE_PX], dtype=float)
            ),
        )
    else:
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, PROJECTION_NOISE_PX)
        stereo_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([PROJECTION_NOISE_PX, PROJECTION_NOISE_PX, PROJECTION_NOISE_PX], dtype=float)
        )
        
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(POSE_PRIOR_SIGMAS)

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    gt_world_to_first = gtsam.Pose3(sequence_sample.world_poses[0].rotation(), np.zeros(3))
    pose_initials = compute_pose_initializations(gt_world_to_first, sequence_results, len(rectified_frames))
    # sequence_velocities = estimate_sequence_velocities(sequence_sample)
    sequence_velocities = [sequence_sample.imu_measurements[i].world_velocity for i in range(sequence_sample.length)]

    for frame_idx in frames_for_ba:
        initial_pose = clone_pose(pose_initials[frame_idx])
        values.insert(X(frame_idx), initial_pose)

    imu_available = use_imu and sequence_sample.imu_measurements is not None

    # imu_params = create_preintegration_params() if imu_available else None
    imu_params = gtsam.PreintegrationCombinedParams.MakeSharedU(IMU_GRAVITY_MAGNITUDE)
    imu_params.n_gravity = IMU_GRAVITY_VECTOR
    imu_params.setAccelerometerCovariance(np.eye(3) * (IMU_ACCEL_NOISE**2))     # e.g. 2e-3^2
    imu_params.setGyroscopeCovariance(np.eye(3) * (IMU_GYRO_NOISE**2))      # e.g. 1.7e-4^2
    imu_params.setIntegrationCovariance(np.eye(3) * (IMU_INTEGRATION_NOISE**2))                     # not ~0
    imu_params.setBiasAccCovariance(np.eye(3) * (IMU_BIAS_PRIOR_SIGMAS[0]**2))                # RW densities
    imu_params.setBiasOmegaCovariance(np.eye(3) * (IMU_BIAS_PRIOR_SIGMAS[3]**2))


    imu_bias_key = None
    imu_bias_initial = gtsam.imuBias.ConstantBias()
    bias_before_vec: np.ndarray | None = None
    bias_after_vec: np.ndarray | None = None

    if imu_available:
        velocity_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, IMU_VELOCITY_PRIOR_SIGMA)
        bias_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(IMU_BIAS_PRIOR_SIGMAS)
        imu_bias_key = B(0)
        if not values.exists(imu_bias_key):
            values.insert(imu_bias_key, imu_bias_initial)

        # # TODO: don't use gt velocities for initialization
        # for frame_idx in frames_for_ba:
        #     vel = (
        #         sequence_velocities[frame_idx]
        #     )
        #     values.insert(V(frame_idx), vel)
        # first_velocity = (
        #     sequence_velocities[frames_for_ba[0]]
        # )
        # graph.add(gtsam.PriorFactorVector(V(frames_for_ba[0]), first_velocity, velocity_prior_noise))
        # graph.add(gtsam.PriorFactorConstantBias(imu_bias_key, imu_bias_initial, bias_prior_noise))

        # only use gt for first velocity
        first_velocity = sequence_velocities[frames_for_ba[0]]
        values.insert(V(frames_for_ba[0]), first_velocity)
        graph.add(gtsam.PriorFactorVector(V(frames_for_ba[0]), first_velocity, velocity_prior_noise))
        # graph.add(gtsam.PriorFactorConstantBias(imu_bias_key, imu_bias_initial, bias_prior_noise))

        # initialize the rest of the velocities by differeniating the poses
        for prev_frame, next_frame in zip(frames_for_ba[:-1], frames_for_ba[1:]):
            assert next_frame > prev_frame

            prev_pose = values.atPose3(X(prev_frame))
            next_pose = values.atPose3(X(next_frame))

            prev_ts = sequence_sample.frame_timestamps[prev_frame]
            next_ts = sequence_sample.frame_timestamps[next_frame]
            dt = next_ts - prev_ts
            assert dt > 0.0 and np.isfinite(dt)

            velocity = (next_pose.translation() - prev_pose.translation()) / dt
            values.insert(V(next_frame), velocity)
        
        # # add a prior for the bias
        # graph.add(gtsam.PriorFactorConstantBias(imu_bias_key, imu_bias_initial, bias_prior_noise))

    graph.add(gtsam.PriorFactorPose3(X(0), gt_world_to_first, prior_noise))

    if imu_available and imu_params is not None and imu_bias_key is not None:
        graph.add(gtsam.PriorFactorConstantBias(
        B(frames_for_ba[0]),
        gtsam.imuBias.ConstantBias(),
        gtsam.noiseModel.Diagonal.Sigmas(
            IMU_BIAS_PRIOR_SIGMAS)))

        for k in frames_for_ba:
            if not values.exists(B(k)):
                values.insert(B(k), gtsam.imuBias.ConstantBias())
            #     if k != frames_for_ba[0]:
            #         graph.add(gtsam.PriorFactorConstantBias(B(k), gtsam.imuBias.ConstantBias(), gtsam.noiseModel.Diagonal.Sigmas(
            # IMU_BIAS_PRIOR_SIGMAS)))

        for prev_frame, next_frame in zip(frames_for_ba[:-1], frames_for_ba[1:]):
            # print(f'Adding imu factor for {prev_frame} and {next_frame}')
            assert next_frame == prev_frame + 1

            # pim = gtsam.PreintegratedImuMeasurements(imu_params, imu_bias_initial)
            pim = gtsam.PreintegratedCombinedMeasurements(imu_params, values.atConstantBias(B(prev_frame)))
            batch = sequence_sample.imu_measurements[next_frame]
            _integrate_imu_batch(pim, batch)
            # pim.integrateMeasurement(np.zeros(3), np.zeros(3), 0.1) # constant velocity

            assert abs(pim.deltaTij() - (sequence_sample.frame_timestamps[next_frame] - sequence_sample.frame_timestamps[prev_frame])) <= 1e-4, f"IMU Δt {pim.deltaTij():.6f} vs expected {sequence_sample.frame_timestamps[next_frame] - sequence_sample.frame_timestamps[prev_frame]:.6f}"

            # pim = preintegrate_between_frames(
            #     sequence_sample,
            #     prev_frame,
            #     next_frame,
            #     imu_params,
            #     imu_bias_initial,
            # )
            graph.add(
                gtsam.CombinedImuFactor(
                    X(prev_frame),
                    V(prev_frame),
                    X(next_frame),
                    V(next_frame),
                    B(prev_frame),
                    B(next_frame),
                    pim,
                )
            )

    # candidate_offsets = np.linspace(-0.02, 0.02, 41)  # -20 ms .. +20 ms
    # best_tau, best_err = 0.0, float("inf")

    # for tau in candidate_offsets:
    #     total = 0.0
    #     for i, j in zip(frames_for_ba[:-1], frames_for_ba[1:]):
    #         pim = preintegrate_between_frames(sequence_sample, i, j, imu_params, imu_bias_initial, time_offset=tau)
    #         f = gtsam.ImuFactor(X(i), V(i), X(j), V(j), B(0), pim)
    #         total += float(f.error(values))
    #     print(f"tau={tau*1e3:+5.1f} ms  IMU-chain error={total:.6f}")
    #     if total < best_err:
    #         best_err, best_tau = total, tau

    # print("Best τ:", best_tau, "s  (IMU-chain error:", best_err, ")")

    if imu_available and imu_bias_key is not None and values.exists(imu_bias_key):
        bias_before_vec = imu_bias_to_numpy(values.atConstantBias(imu_bias_key))

    landmark_keys: list[int] = []
    landmark_index_lookup: list[int] = []
    initial_landmarks: list[np.ndarray] = []
    observations: list[tuple[int, int, np.ndarray]] = []
    stereo_counts = {frame_idx: 0 for frame_idx in frames_for_ba}
    mono_counts = {frame_idx: 0 for frame_idx in frames_for_ba}

    inlier_lookup: dict[int, set[int]] = {}
    for result in sequence_results:
        if result.get("status") != "success":
            continue
        frame_idx = result["frame_index"]
        track_ids = result.get("track_ids")
        if track_ids is None:
            continue
        matches = result["matched_pair"].matches
        if matches.size == 0:
            continue
        matched_track_ids = np.asarray(track_ids, dtype=int)
        valid_ids = matched_track_ids[matches[:, 0]]
        inlier_lookup.setdefault(frame_idx, set()).update(valid_ids.tolist())

    for track_id in sorted(tracks.keys()):
        track = tracks[track_id]
        if not np.isfinite(track.anchor_depth) or track.anchor_depth <= 0.0:
            continue
        if track.anchor_frame not in frames_for_ba:
            continue

        anchor_pose = pose_initials[track.anchor_frame]
        anchor_point = gtsam.Point3(*track.anchor_point3.tolist())
        world_point = anchor_pose.transformFrom(anchor_point)
        world_point_np = point3_like_to_numpy(world_point)

        observation_frames: list[tuple[int, TrackObservation]] = []
        for frame_idx in frames_for_ba:
            observation = track_history[frame_idx].get(track_id)
            if observation is None:
                continue
            if (
                USE_INLIER_OBSERVATIONS_ONLY
                and frame_idx != track.anchor_frame
                and frame_idx in inlier_lookup
                and track_id not in inlier_lookup[frame_idx]
            ):
                continue

            pose_initial = pose_initials[frame_idx]
            camera_initial = gtsam.PinholeCameraCal3_S2(pose_initial, calibration)
            try:
                predicted = camera_initial.project(world_point)
            except RuntimeError:
                continue
            predicted_vec = predicted.vector() if hasattr(predicted, "vector") else np.asarray(predicted)
            measurement_vec = np.asarray(observation.keypoint, dtype=np.float64)
            residual = predicted_vec - measurement_vec
            if np.linalg.norm(residual) > REPROJECTION_GATING_THRESHOLD_PX:
                continue

            observation_frames.append((frame_idx, observation))

        if len(observation_frames) < MIN_OBSERVATIONS_PER_LANDMARK:
            continue

        landmark_key = len(landmark_keys)
        landmark_keys.append(landmark_key)
        landmark_index_lookup.append(track_id)

        values.insert(L(landmark_key), world_point)
        initial_landmarks.append(world_point_np)

        # # add prior to enforce pseduo motion only BA
        # ma_factor = gtsam.PriorFactorPoint3(
        #     L(landmark_key),
        #     world_point,
        #     gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0], dtype=float) * 0.00001),
        # )
        # graph.add(ma_factor)

        for frame_idx, observation in observation_frames:
            measurement_vec = np.asarray(observation.keypoint, dtype=np.float64)
            if frame_idx == track.anchor_frame:
                depth_value = float(track.anchor_depth)
                if not np.isfinite(depth_value) or depth_value <= 0.0:
                    continue
                fx = stereo_calibration.fx()
                baseline = stereo_calibration.baseline()
                disparity = (fx * baseline) / depth_value
                stereo_measurement = gtsam.StereoPoint2(
                    float(track.anchor_keypoint[0]),
                    float(track.anchor_keypoint[0] - disparity),
                    float(track.anchor_keypoint[1]),
                )
                graph.add(
                    gtsam.GenericStereoFactor3D(
                        stereo_measurement,
                        stereo_noise,
                        X(frame_idx),
                        L(landmark_key),
                        stereo_calibration,
                    )
                )
                stereo_counts[frame_idx] += 1
            else:
                depth_value = float(observation.depth)
                # if False:
                if np.isfinite(depth_value) and depth_value > 0.0:
                    fx = stereo_calibration.fx()
                    baseline = stereo_calibration.baseline()
                    disparity = (fx * baseline) / depth_value
                    stereo_measurement = gtsam.StereoPoint2(
                        float(observation.keypoint[0]),
                        float(observation.keypoint[0] - disparity),
                        float(observation.keypoint[1]),
                    )
                    graph.add(
                        gtsam.GenericStereoFactor3D(
                            stereo_measurement,
                            stereo_noise,
                            X(frame_idx),
                            L(landmark_key),
                            stereo_calibration,
                        )
                    )
                    stereo_counts[frame_idx] += 1
                else:
                    graph.add(
                        gtsam.GenericProjectionFactorCal3_S2(
                            gtsam.Point2(float(measurement_vec[0]), float(measurement_vec[1])),
                            measurement_noise,
                            X(frame_idx),
                            L(landmark_key),
                            calibration,
                        )
                    )
                    mono_counts[frame_idx] += 1

            observations.append((frame_idx, landmark_key, measurement_vec))

    if not observations:
        raise RuntimeError("No landmark observations available for bundle adjustment.")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values)
    optimized_values = optimizer.optimize()

    if USE_IMU_FACTORS:
        def sum_vision_error(graph, values):
            total = 0.0
            for idx in range(graph.size()):
                f = graph.at(idx)
                name = type(f).__name__
                if "ProjectionFactor" in name or "StereoFactor" in name:
                    total += float(f.error(values))
            return total

        def sum_imu_error(graph, values):
            total = 0.0
            for idx in range(graph.size()):
                f = graph.at(idx)
                name = type(f).__name__
                if "ImuFactor" in name:
                    total += float(f.error(values))
            return total

        vision_err = sum_vision_error(graph, values)
        imu_err = sum_imu_error(graph, values)

        print(f"[diag] Vision error evaluated on IMU-only poses: {vision_err:.6f}")
        print(f"[diag] IMU error evaluated on IMU-only poses: {imu_err:.6f}")

    optimized_pose_dict: dict[int, gtsam.Pose3] = {}
    for frame_idx in frames_for_ba:
        optimized_pose_dict[frame_idx] = optimized_values.atPose3(X(frame_idx))

    optimized_landmarks: list[np.ndarray] = []
    for key in range(len(landmark_keys)):
        if hasattr(optimized_values, "atPoint3"):
            point = optimized_values.atPoint3(L(key))
        else:
            point = optimized_values.atVector(L(key))
        optimized_landmarks.append(point3_like_to_numpy(point))

    initial_pose_dict = {idx: values.atPose3(X(idx)) for idx in frames_for_ba}

    observation_matrix = [(frame_idx, landmark_idx, measurement) for frame_idx, landmark_idx, measurement in observations]
    # observation_matrix = []

    reprojection_before = compute_reprojection_errors(initial_pose_dict, initial_landmarks, observation_matrix, calibration)
    reprojection_after = compute_reprojection_errors(optimized_pose_dict, optimized_landmarks, observation_matrix, calibration)
    # reprojection_before = np.zeros(len(frames_for_ba))
    # reprojection_after = np.zeros(len(frames_for_ba))

    if imu_available and imu_bias_key is not None and optimized_values.exists(imu_bias_key):
        bias_after_vec = imu_bias_to_numpy(optimized_values.atConstantBias(imu_bias_key))
        if bias_before_vec is not None:
            log_imu_bias_values(bias_before_vec, bias_after_vec, label=log_prefix)
            plot_imu_bias_values(bias_before_vec, bias_after_vec, label=log_prefix)

    return {
        "frames_for_ba": frames_for_ba,
        "landmark_original_indices": landmark_index_lookup,
        "initial_pose_dict": initial_pose_dict,
        "optimized_pose_dict": optimized_pose_dict,
        "initial_landmarks": initial_landmarks,
        "optimized_landmarks": optimized_landmarks,
        "reprojection_before": reprojection_before,
        "reprojection_after": reprojection_after,
        "stereo_counts": stereo_counts,
        "mono_counts": mono_counts,
        "imu_bias_before": bias_before_vec,
        "imu_bias_after": bias_after_vec,
    }


# is this right?? I just want percent error over WHOLE trajectory, not average cummulative error? maybe that means I should just look at the end of the graph or at the dots? idk
# I want to calculate rmse over whole trajectory basically
def compute_pose_errors_against_ground_truth(
    optimized_pose_dict: dict[int, gtsam.Pose3],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
) -> dict[int, dict[str, np.ndarray]]:
    gt_distance = 0.0
    prev_pose = None
    errors: dict[int, dict[str, np.ndarray]] = {}
    for frame_idx, pose in optimized_pose_dict.items():
        gt_pose = gtsam.Pose3(gtsam.Rot3.Identity(), sequence_sample.world_poses[0].translation()).inverse() * sequence_sample.world_poses[frame_idx]
        gt_translation = to_numpy_vec3(gt_pose.translation())

        # gt_distance = float(np.linalg.norm(gt_translation))
        if prev_pose is None:
            prev_pose = pose
            gt_distance = float(np.linalg.norm(gt_translation)) # is this starting at the origin correctly so this will be valid?
        else:
            gt_distance += float(np.linalg.norm(gt_translation - prev_pose.translation()))
            prev_pose = pose

        error = gt_pose.between(pose)
        trans, rot = pose_error_components(error)
        translation_norm = float(np.linalg.norm(trans))
        translation_error_pct = float(np.nan)
        if gt_distance > 1e-6:
            translation_error_pct = float((translation_norm / gt_distance) * 100.0)
        errors[frame_idx] = {
            "translation": trans,
            "rotation": rot,
            "translation_norm": translation_norm,
            "rotation_norm_rad": np.linalg.norm(rot),
            "ground_truth_distance": gt_distance,
            "translation_error_pct": translation_error_pct,
        }
    return errors


# def distance_percentage_series(
#     pose_errors: dict[int, dict[str, np.ndarray]],
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     distances: list[float] = []
#     translations: list[np.ndarray] = []
#     percentages: list[float] = []
#     for frame_idx in sorted(pose_errors):
#         info = pose_errors[frame_idx]
#         trans = info.get("translation", np.nan)
#         pct = float(info.get("translation_error_pct", np.nan))
#         dist = float(info.get("ground_truth_distance", np.nan))
#         if not np.isfinite(pct) or not np.isfinite(dist):
#             continue
#         distances.append(dist)
#         translations.append(trans)
#         percentages.append(pct)

#     if not distances:
#         empty = np.asarray([], dtype=np.float64)
#         return empty, empty, empty

#     distances_arr = np.asarray(distances, dtype=np.float64)
#     percentages_arr = np.asarray(percentages, dtype=np.float64)
#     translations_arr = np.asarray(translations, dtype=np.float64)
#     order = np.argsort(distances_arr)
#     distances_sorted = distances_arr[order]
#     percentages_sorted = percentages_arr[order]
#     translations_sorted = translations_arr[order]
#     cumulative_mean = np.cumsum(percentages_sorted) / np.arange(1, percentages_sorted.size + 1, dtype=np.float64)

#     return distances_sorted, percentages_sorted, cumulative_mean

def distance_percentage_series(
    pose_errors: dict[int, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances: list[float] = []
    translations: list[np.ndarray] = []
    percentages: list[float] = []

    for frame_idx in sorted(pose_errors):
        info = pose_errors[frame_idx]
        trans = info.get("translation", np.nan)
        pct = float(info.get("translation_error_pct", np.nan))
        dist = float(info.get("ground_truth_distance", np.nan))
        if not np.isfinite(dist) or not np.isfinite(pct):
            continue
        distances.append(dist)
        translations.append(trans)
        percentages.append(pct)

    if not distances:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty

    distances_arr = np.asarray(distances, dtype=np.float64)
    translations_arr = np.asarray(translations, dtype=np.float64)  # shape (N, 3)
    percentages_arr = np.asarray(percentages, dtype=np.float64)

    # Sort by ground-truth distance
    order = np.argsort(distances_arr)
    distances_sorted = distances_arr[order]
    translations_sorted = translations_arr[order]
    percentages_sorted = percentages_arr[order]

    # Per-frame translation norm
    trans_norms = np.linalg.norm(translations_sorted, axis=1)  # (N,)

    # Cumulative RMSE of translation error up to each point
    sq_err = trans_norms ** 2
    cum_sq_err = np.cumsum(sq_err)
    counts = np.arange(1, len(trans_norms) + 1, dtype=np.float64)
    rmse = np.sqrt(cum_sq_err / counts)  # (N,)

    # Cumulative "percent drift" = RMSE / total distance so far
    cumulative_mean = np.full_like(rmse, np.nan)
    valid = distances_sorted > 1e-9
    cumulative_mean[valid] = (rmse[valid] / distances_sorted[valid]) * 100.0

    return distances_sorted, percentages_sorted, cumulative_mean


def save_results(
    tracking_summary: dict[str, Any],
    ba_result: dict[str, Any],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
) -> None:
    frame_ids = np.array(sequence_sample.frame_ids, dtype=int)
    frame_indices = np.array(sequence_sample.frame_indices, dtype=int)

    optimized_pose_dict = ba_result["optimized_pose_dict"]
    pose_errors = compute_pose_errors_against_ground_truth(optimized_pose_dict, sequence_sample)

    pose_error_frames = sorted(pose_errors.keys())
    optimized_translation_errors = np.array(
        [pose_errors[idx]["translation"] for idx in pose_error_frames],
        dtype=np.float64,
    )
    optimized_rotation_errors = np.array(
        [pose_errors[idx]["rotation"] for idx in pose_error_frames],
        dtype=np.float64,
    )

    np.savez(
        RESULTS_PATH,
        frame_ids=frame_ids,
        frame_indices=frame_indices,
        tracking_summary=tracking_summary,
        frames_for_ba=np.array(ba_result["frames_for_ba"], dtype=int),
        landmark_original_indices=np.array(ba_result["landmark_original_indices"], dtype=int),
        reprojection_before=ba_result["reprojection_before"],
        reprojection_after=ba_result["reprojection_after"],
        optimized_translation_errors=optimized_translation_errors,
        optimized_rotation_errors=optimized_rotation_errors,
        optimized_translation_norms=np.linalg.norm(optimized_translation_errors, axis=1),
        optimized_rotation_norms_deg=np.rad2deg(np.linalg.norm(optimized_rotation_errors, axis=1)),
        stereo_counts=np.array([ba_result["stereo_counts"][idx] for idx in ba_result["frames_for_ba"]], dtype=int),
        mono_counts=np.array([ba_result["mono_counts"][idx] for idx in ba_result["frames_for_ba"]], dtype=int),
    )
    print(f"Saved bundle adjustment results to {RESULTS_PATH}")


def plot_feature_tracks(
    rectified_frames: list[RectifiedStereoFrame],
    track_history: list[dict[int, TrackObservation]],
    tracks: dict[int, FeatureTrack],
) -> None:
    anchor_image = rectified_frames[0].left_rect
    anchor_points = (
        np.asarray([obs.keypoint for obs in track_history[0].values()], dtype=np.float32)
        if track_history and track_history[0]
        else np.empty((0, 2), dtype=np.float32)
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].set_title("Frame 0 features")
    axes[0].imshow(anchor_image)
    if anchor_points.size:
        axes[0].scatter(anchor_points[:, 0], anchor_points[:, 1], s=10, c="lime", linewidths=0.5)
    axes[0].axis("off")

    last_frame_image = rectified_frames[-1].left_rect
    axes[1].set_title("Tracked KLT trajectories")
    axes[1].imshow(last_frame_image)
    cmap = plt.cm.get_cmap("viridis", max(1, len(tracks)))
    for color_idx, track_id in enumerate(sorted(tracks.keys())):
        track = tracks[track_id]
        if len(track.observations) < 2:
            continue
        frames = sorted(track.observations.keys())
        path = np.asarray([track.observations[f].keypoint for f in frames], dtype=np.float32)
        axes[1].plot(path[:, 0], path[:, 1], color=cmap(color_idx % cmap.N), linewidth=1.0)
        axes[1].scatter(path[-1, 0], path[-1, 1], s=8, color=cmap(color_idx % cmap.N))
    axes[1].axis("off")

    frame_indices = list(range(len(track_history)))
    track_counts = [len(frame_obs) for frame_obs in track_history]

    axes[2].set_title("Per-frame track counts")
    axes[2].plot(frame_indices, track_counts, marker="o")
    axes[2].set_xlabel("Frame index")
    axes[2].set_ylabel("Visible tracks")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_match_debug(
    rectified_frames: list[RectifiedStereoFrame],
    results: list[dict[str, Any]],
) -> None:
    successful = [res for res in results if res["status"] == "success"]
    if not successful:
        return

    highlight = successful[-1]
    frame_idx = highlight["frame_index"]
    prev_idx = max(0, frame_idx - 1)
    pair = FramePair(first=rectified_frames[prev_idx], second=rectified_frames[frame_idx])

    mkpts1 = highlight["matched_pair"].mkpts1
    mkpts2 = highlight["matched_pair"].mkpts2
    if hasattr(mkpts1, "cpu"):
        mkpts1 = mkpts1.cpu().numpy()
    if hasattr(mkpts2, "cpu"):
        mkpts2 = mkpts2.cpu().numpy()

    image = draw_matches(
        pair,
        mkpts1,
        mkpts2,
    )

    plt.figure(figsize=(12, 6))
    plt.title(f"KLT correspondences for frame {frame_idx}")
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def plot_imu_measurements(sequence_sample: test_utils.FrameSequenceWithGroundTruth, title: str | None = None) -> None:
    imu_batches = sequence_sample.imu_measurements

    timestamps_list: list[np.ndarray] = []
    accelerations: list[np.ndarray] = []
    gyroscopes: list[np.ndarray] = []

    for batch in imu_batches:
        if batch.timestamps.size == 0:
            continue
        timestamps_list.append(batch.timestamps)
        accelerations.append(batch.linear_accelerations)
        gyroscopes.append(batch.angular_velocities)

    if not timestamps_list:
        return

    times = np.concatenate(timestamps_list, axis=0)
    acc = np.concatenate(accelerations, axis=0)
    gyro = np.concatenate(gyroscopes, axis=0)

    if times.size == 0:
        return
    base_time = float(times[0])
    times = times - base_time

    component_labels = ["X", "Y", "Z"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(title or "IMU measurements", fontsize=14)

    for idx in range(3):
        axes[0].plot(times, acc[:, idx], label=f"acc {component_labels[idx]}")
        axes[1].plot(times, gyro[:, idx], label=f"gyro {component_labels[idx]}")

    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_ylabel("Angular velocity (rad/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    ypr_axis = axes[2]
    frame_times = np.array([batch.frame_timestamp for batch in imu_batches], dtype=np.float64)
    frame_times = frame_times - float(frame_times[0])
    ypr_axis.set_xlabel("Time (s)")
    ypr_axis.set_ylabel("Angle (deg)")
    ypr_axis.set_title("Orientation (Y/P/R)")

    inertial_poses: list[gtsam.Pose3] | None = None
    inertial_poses = integrate_inertial_trajectory(sequence_sample)

    if inertial_poses:
        gt_ypr = np.array(
            [np.asarray(pose.rotation().ypr(), dtype=np.float64) for pose in sequence_sample.world_poses],
            dtype=np.float64,
        )
        imu_ypr = np.array(
            [np.asarray(pose.rotation().ypr(), dtype=np.float64) for pose in inertial_poses],
            dtype=np.float64,
        )
        count = min(frame_times.shape[0], gt_ypr.shape[0], imu_ypr.shape[0])
        frame_times_plot = frame_times[:count]
        gt_ypr_deg = np.rad2deg(gt_ypr[:count])
        imu_ypr_deg = np.rad2deg(imu_ypr[:count])
        ypr_labels = ["Yaw", "Pitch", "Roll"]
        colors = ["tab:blue", "tab:orange", "tab:green"]

        for idx, label in enumerate(ypr_labels):
            color = colors[idx % len(colors)]
            ypr_axis.plot(frame_times_plot, gt_ypr_deg[:, idx], color=color, linestyle="-", label=f"GT {label}")
            ypr_axis.plot(frame_times_plot, imu_ypr_deg[:, idx], color=color, linestyle="--", label=f"IMU {label}")
        ypr_axis.legend(ncol=2)
    else:
        ypr_axis.text(
            0.5,
            0.5,
            "IMU integration unavailable",
            transform=ypr_axis.transAxes,
            ha="center",
            va="center",
        )
    ypr_axis.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    mean_ypr_end_error = np.mean(np.abs(imu_ypr_deg[-1] - gt_ypr_deg[-1]))
    mean_ypr_error_per_sec = mean_ypr_end_error / (frame_times[-1] - frame_times[0])
    print(f"Mean YPR end error: {mean_ypr_end_error:.3f} deg ({mean_ypr_error_per_sec * 3600:.3f} deg/hr)")


def log_imu_bias_values(
    bias_before: np.ndarray,
    bias_after: np.ndarray,
    label: str | None = None,
) -> None:
    if bias_before.size != 6 or bias_after.size != 6:
        return
    prefix = f"[{label}] " if label else ""
    before_acc = bias_before[:3]
    before_gyro = bias_before[3:]
    after_acc = bias_after[:3]
    after_gyro = bias_after[3:]
    print(f"{prefix}IMU bias before BA: acc={before_acc.tolist()}, gyro={before_gyro.tolist()}")
    print(f"{prefix}IMU bias after BA:  acc={after_acc.tolist()}, gyro={after_gyro.tolist()}")


def plot_imu_bias_values(
    bias_before: np.ndarray,
    bias_after: np.ndarray,
    label: str | None = None,
) -> None:
    if bias_before.size != 6 or bias_after.size != 6:
        return
    x = np.arange(6)
    component_labels = ["acc X", "acc Y", "acc Z", "gyro X", "gyro Y", "gyro Z"]
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, bias_before, width, label="Before BA")
    ax.bar(x + width / 2, bias_after, width, label="After BA")
    ax.set_xticks(x)
    ax.set_xticklabels(component_labels, rotation=45, ha="right")
    ax.set_ylabel("Bias value")
    title_prefix = f"{label} " if label else ""
    ax.set_title(f"{title_prefix}IMU bias comparison")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_pose_trajectories(
    initial_pose_dict: dict[int, gtsam.Pose3],
    optimized_pose_dict: dict[int, gtsam.Pose3],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
    initial_landmarks: list[np.ndarray] | None = None,
    optimized_landmarks: list[np.ndarray] | None = None,
    landmark_stride: int = 1,
    inertial_poses: list[gtsam.Pose3] | None = None,
    optimized_pose_dict_no_imu: dict[int, gtsam.Pose3] | None = None,
    initial_label: str = "Initial PnP",
    optimized_label: str = "Optimized BA",
    additional_pose_dicts: dict[str, dict[int, gtsam.Pose3]] | None = None,
    additional_initial_pose_dicts: dict[str, dict[int, gtsam.Pose3]] | None = None,
) -> None:
    frames = sorted(optimized_pose_dict.keys())

    base_frame = frames[0]
    base_initial = initial_pose_dict[base_frame]
    base_optimized = optimized_pose_dict[base_frame]

    initial_xyz = np.array(
        [
            to_numpy_vec3((initial_pose_dict[f]).translation())
            for f in frames
        ],
        dtype=np.float64,
    )
    optimized_xyz = np.array(
        [
            to_numpy_vec3((optimized_pose_dict[f]).translation())
            for f in frames
        ],
        dtype=np.float64,
    )
    ground_truth_xyz = np.array(
        [to_numpy_vec3((gtsam.Pose3(gtsam.Rot3.Identity(), sequence_sample.world_poses[0].translation()).inverse() * sequence_sample.world_poses[f]).translation()) for f in frames],
        dtype=np.float64,
    )
    optimized_xyz_no_imu: np.ndarray | None = None
    if optimized_pose_dict_no_imu is not None:
        optimized_xyz_no_imu = np.array(
            [
                to_numpy_vec3(
                    (optimized_pose_dict_no_imu[f]).translation()
                )
                for f in frames
            ],
            dtype=np.float64,
        )

    inertial_xyz: np.ndarray | None = None
    if inertial_poses:
        try:
            # inertial_points: list[np.ndarray] = []
            # for idx, frame in enumerate(frames):
            #     if frame >= len(inertial_poses):
            #         break
            #     imu_pose = inertial_poses[frame]
            #     inertial_points.append(imu_pose.translation())
            # if inertial_points:
                # inertial_xyz = np.asarray(inertial_points, dtype=np.float64)
            inertial_xyz = np.array([imu_pose.translation() for imu_pose in inertial_poses])
        except IndexError:
            print("IMU trajectory shorter than pose set; skipping inertial overlay.")
            inertial_xyz = None

    plt.figure(figsize=(10, 6))
    plt.title("Trajectory comparison (XY plane)")
    plt.plot(initial_xyz[:, 0], initial_xyz[:, 1], "o--", label=initial_label)
    plt.plot(optimized_xyz[:, 0], optimized_xyz[:, 1], "o-", label=optimized_label)
    if optimized_xyz_no_imu is not None:
        plt.plot(optimized_xyz_no_imu[:, 0], optimized_xyz_no_imu[:, 1], "o-", label="BA w/o IMU")
    if additional_initial_pose_dicts:
        for label, pose_dict in additional_initial_pose_dicts.items():
            if base_frame not in pose_dict:
                continue
            base_pose = pose_dict[base_frame]
            additional_points: list[np.ndarray] = []
            for frame in frames:
                pose = pose_dict.get(frame)
                if pose is None:
                    additional_points = []
                    break
                relative_pose = pose
                additional_points.append(to_numpy_vec3(relative_pose.translation()))
            if not additional_points:
                continue
            additional_xyz = np.asarray(additional_points, dtype=np.float64)
            plt.plot(additional_xyz[:, 0], additional_xyz[:, 1], "o--", label=label)
    if additional_pose_dicts:
        for label, pose_dict in additional_pose_dicts.items():
            if base_frame not in pose_dict:
                continue
            base_pose = pose_dict[base_frame]
            additional_points = []
            for frame in frames:
                if frame not in pose_dict:
                    continue
                relative_pose = pose_dict[frame]
                additional_points.append(to_numpy_vec3(relative_pose.translation()))
            if not additional_points:
                continue
            additional_xyz = np.asarray(additional_points, dtype=np.float64)
            plt.plot(additional_xyz[:, 0], additional_xyz[:, 1], "o-", label=label)
    if inertial_xyz is not None:
        plt.plot(
            inertial_xyz[:, 0],
            inertial_xyz[:, 1],
            "s--",
            label="IMU-only (per-frame)",
        )
    plt.plot(ground_truth_xyz[:, 0], ground_truth_xyz[:, 1], "x-", label="Ground truth")

    if initial_landmarks:
        init_landmarks_arr = np.asarray(initial_landmarks, dtype=np.float64)
        if landmark_stride > 1:
            init_landmarks_arr = init_landmarks_arr[::landmark_stride]
        plt.scatter(
            init_landmarks_arr[:, 0],
            init_landmarks_arr[:, 1],
            s=6,
            c="#66c2a5",
            alpha=0.35,
            label="Landmarks before BA",
        )
    if optimized_landmarks:
        opt_landmarks_arr = np.asarray(optimized_landmarks, dtype=np.float64)
        if landmark_stride > 1:
            opt_landmarks_arr = opt_landmarks_arr[::landmark_stride]
        plt.scatter(
            opt_landmarks_arr[:, 0],
            opt_landmarks_arr[:, 1],
            s=6,
            c="#fc8d62",
            alpha=0.35,
            label="Landmarks after BA",
        )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.show()


def plot_pose_error_norms_per_frame(
    initial_pose_dict: dict[int, gtsam.Pose3],
    optimized_pose_dict: dict[int, gtsam.Pose3],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
    comparison_pose_dict: dict[int, gtsam.Pose3] | None = None,
) -> None:
    shared_frames = sorted(set(initial_pose_dict.keys()) & set(optimized_pose_dict.keys()))
    if not shared_frames:
        return

    initial_errors = compute_pose_errors_against_ground_truth(initial_pose_dict, sequence_sample)
    optimized_errors = compute_pose_errors_against_ground_truth(optimized_pose_dict, sequence_sample)
    comparison_errors = (
        compute_pose_errors_against_ground_truth(comparison_pose_dict, sequence_sample)
        if comparison_pose_dict is not None
        else None
    )

    translation_before = [initial_errors[frame_idx]["translation_norm"] for frame_idx in shared_frames]
    translation_after = [optimized_errors[frame_idx]["translation_norm"] for frame_idx in shared_frames]
    rotation_before_deg = [
        np.rad2deg(initial_errors[frame_idx]["rotation_norm_rad"]) for frame_idx in shared_frames
    ]
    rotation_after_deg = [
        np.rad2deg(optimized_errors[frame_idx]["rotation_norm_rad"]) for frame_idx in shared_frames
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].set_title("Translation error norms")
    # axes[0].plot(shared_frames, translation_before, "o--", label="Before BA")
    axes[0].plot(shared_frames, translation_after, "o-", label="After BA (IMU)")
    if comparison_errors is not None:
        translation_comp = [comparison_errors[idx]["translation_norm"] for idx in shared_frames]
        axes[0].plot(shared_frames, translation_comp, "o-", label="After BA (no IMU)")
    axes[0].set_xlabel("Frame index")
    axes[0].set_ylabel("Translation error (m)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_title("Rotation error norms")
    # axes[1].plot(shared_frames, rotation_before_deg, "o--", label="Before BA")
    axes[1].plot(shared_frames, rotation_after_deg, "o-", label="After BA (IMU)")
    if comparison_errors is not None:
        rotation_comp_deg = [
            np.rad2deg(comparison_errors[idx]["rotation_norm_rad"]) for idx in shared_frames
        ]
        axes[1].plot(shared_frames, rotation_comp_deg, "o-", label="After BA (no IMU)")
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel("Rotation error (deg)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_reprojection_error_histograms(
    reprojection_before: np.ndarray,
    reprojection_after: np.ndarray,
) -> None:
    plt.figure(figsize=(10, 5))
    bins = np.linspace(0, max(5.0, float(np.nanmax(reprojection_before))), 40)
    plt.hist(reprojection_before, bins=bins, alpha=0.6, label="Before BA")
    plt.hist(reprojection_after, bins=bins, alpha=0.6, label="After BA")
    plt.xlabel("Reprojection error (px)")
    plt.ylabel("Count")
    plt.title("Reprojection error distribution")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()


def plot_translation_error_percentages(
    series_data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    if not series_data:
        return

    plt.figure(figsize=(10, 5))
    for label, distances, per_frame_pct, cumulative_pct in series_data:
        if distances.size == 0:
            continue
        plt.scatter(
            distances,
            per_frame_pct,
            alpha=0.35,
            s=20,
            label=f"{label} per-frame",
        )
        plt.plot(
            distances,
            cumulative_pct,
            linewidth=2,
            label=f"{label} cumulative mean",
        )

        # plt.plot(
        #     distances,
        #     per_frame_pct,
        #     linewidth=2,
        #     label=f"{label} per-frame",
        # )

    plt.xlabel("Ground-truth distance (m)")
    plt.ylabel("Translation error (% of distance)")
    plt.title("Translation error percentage vs distance")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_track_lengths(tracks: dict[int, FeatureTrack]) -> np.ndarray:
    if not tracks:
        return np.asarray([], dtype=int)
    lengths = [len(track.observations) for track in tracks.values()]
    return np.asarray(lengths, dtype=int)


def plot_track_length_histogram(tracks: dict[int, FeatureTrack]) -> None:
    lengths = compute_track_lengths(tracks)
    if lengths.size == 0:
        return

    plt.figure(figsize=(9, 4))
    bins = np.arange(1, lengths.max() + 2) - 0.5
    plt.hist(lengths, bins=bins, edgecolor="black", alpha=0.7)
    plt.xlabel("Track length (frames)")
    plt.ylabel("Count")
    plt.title("Feature track length distribution")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(np.arange(1, lengths.max() + 1))
    plt.tight_layout()
    plt.show()


def run_depth_variant(
    depth_label: str,
    rectified_frames: list[RectifiedStereoFrame],
    depth_frames: list[StereoDepthFrame],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
) -> dict[str, Any]:
    track_history, tracks = track_features_with_refill(rectified_frames, depth_frames)

    sequence_results = estimate_sequence_poses(
        rectified_frames=rectified_frames,
        depth_frames=depth_frames,
        track_history=track_history,
        sequence_sample=sequence_sample,
    )

    tracking_summary = summarise_tracking_results(sequence_sample, track_history, sequence_results)
    if rerun_logger is not None and rerun_logger.enabled:
        _log_sequence_results_to_rerun(rerun_logger, depth_label, depth_frames, sequence_results)

    ba_result = run_bundle_adjustment(
        rectified_frames=rectified_frames,
        track_history=track_history,
        tracks=tracks,
        sequence_results=sequence_results,
        sequence_sample=sequence_sample,
        use_imu=USE_IMU_FACTORS,
        log_prefix=variant_label,
    )
    if rerun_logger is not None and rerun_logger.enabled:
        _log_bundle_adjustment_to_rerun(
            rerun_logger,
            depth_label,
            depth_frames,
            ba_result,
            "with_imu" if USE_IMU_FACTORS else "vision_only",
        )

    ba_result_no_imu = None
    if USE_IMU_FACTORS:
        ba_result_no_imu = run_bundle_adjustment(
            rectified_frames=rectified_frames,
            track_history=track_history,
            tracks=tracks,
            sequence_results=sequence_results,
            sequence_sample=sequence_sample,
            use_imu=False,
            log_prefix=variant_label,
        )
        if rerun_logger is not None and rerun_logger.enabled and ba_result_no_imu is not None:
            _log_bundle_adjustment_to_rerun(
                rerun_logger,
                depth_label,
                depth_frames,
                ba_result_no_imu,
                "vision_only",
            )

    pose_errors = compute_pose_errors_against_ground_truth(ba_result["optimized_pose_dict"], sequence_sample)
    pose_errors_no_imu = None
    if ba_result_no_imu is not None:
        pose_errors_no_imu = compute_pose_errors_against_ground_truth(
            ba_result_no_imu["optimized_pose_dict"],
            sequence_sample,
        )

    optimized_translation_norms = [pose_errors[idx]["translation_norm"] for idx in sorted(pose_errors)]
    optimized_rotation_norms_deg = [
        np.rad2deg(pose_errors[idx]["rotation_norm_rad"]) for idx in sorted(pose_errors)
    ]
    optimized_translation_norms_no_imu: list[float] = []
    optimized_rotation_norms_deg_no_imu: list[float] = []
    if pose_errors_no_imu is not None:
        optimized_translation_norms_no_imu = [
            pose_errors_no_imu[idx]["translation_norm"] for idx in sorted(pose_errors_no_imu)
        ]
        optimized_rotation_norms_deg_no_imu = [
            np.rad2deg(pose_errors_no_imu[idx]["rotation_norm_rad"]) for idx in sorted(pose_errors_no_imu)
        ]

    rms_before = float(np.sqrt(np.mean(ba_result["reprojection_before"] ** 2)))
    rms_after = float(np.sqrt(np.mean(ba_result["reprojection_after"] ** 2)))

    return {
        "label": depth_label,
        "track_history": track_history,
        "sequence_results": sequence_results,
        "tracking_summary": tracking_summary,
        "ba_result": ba_result,
        "ba_result_no_imu": ba_result_no_imu,
        "initial_pose_dict": ba_result["initial_pose_dict"],
        "optimized_pose_dict": ba_result["optimized_pose_dict"],
        "pose_errors": pose_errors,
        "pose_errors_no_imu": pose_errors_no_imu,
        "optimized_translation_norms": optimized_translation_norms,
        "optimized_rotation_norms_deg": optimized_rotation_norms_deg,
        "optimized_translation_norms_no_imu": optimized_translation_norms_no_imu,
        "optimized_rotation_norms_deg_no_imu": optimized_rotation_norms_deg_no_imu,
        "rms_before": rms_before,
        "rms_after": rms_after,
        "tracks": tracks,
    }


# %%
# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
all_tracking_results: list[list[dict[str, Any]]] = []
all_ba_results: list[dict[str, Any]] = []

for sample_idx in range(NUM_SEQUENCE_SAMPLES):
    seed = BASE_SEED + sample_idx
    sequence = test_utils.load_tartanair_sequence_segment(
        env=ENVIRONMENT,
        difficulty=DIFFICULTY,
        traj=TRAJECTORY,
        sequence_length=SEQUENCE_LENGTH,
        seed=seed,
        sampling_mode=SAMPLING_MODE,
        min_stride=MIN_STRIDE,
        max_stride=MAX_STRIDE,
        load_ground_truth_depth=USE_GROUND_TRUTH_DEPTH,
        add_imu_noise=True
    )
    # sequence = test_utils.load_euroc_sequence_segment(
    #     seq_name="MH_01_easy",
    #     sequence_length=SEQUENCE_LENGTH,
    #     seed=seed,
    # )
    plot_imu_measurements(sequence, title=f"IMU data (sample {sample_idx})")
    inertial_poses = integrate_inertial_trajectory(sequence)

    # rectified_frames = [frame.rectify() for frame in sequence.frames]
    rectified_frames = [
        RectifiedStereoFrame(
            left=frame.left,
            right=frame.right,
            left_rect=frame.left,
            right_rect=frame.right,
            calibration=frame.calibration,
        )
        for frame in sequence.frames
    ]
    depth_variants: list[tuple[str, list[StereoDepthFrame]]] = []
    if USE_GROUND_TRUTH_DEPTH:
        if sequence.ground_truth_depths is None:
            raise RuntimeError("Ground-truth depth requested but not provided by the sequence loader.")
        gt_depth_frames = build_ground_truth_depth_frames(
            rectified_frames,
            sequence.ground_truth_depths,
            max_depth=MAX_DEPTH,
        )
        depth_variants.append(("ground_truth", gt_depth_frames))
        sgbm_depth_frames = [sgbm.compute_depth(frame, max_depth=MAX_DEPTH) for frame in rectified_frames]
        depth_variants.append(("sgbm", sgbm_depth_frames))
    else:
        depth_variants.append(
            ("sgbm", [sgbm.compute_depth(frame, max_depth=MAX_DEPTH) for frame in rectified_frames])
        )

    variant_results: list[dict[str, Any]] = []
    for variant_label, variant_depth_frames in depth_variants:
        print(f"Running VO/BA with {variant_label} depth...")
        variant_results.append(
            run_depth_variant(
                depth_label=variant_label,
                rectified_frames=rectified_frames,
                depth_frames=variant_depth_frames,
                sequence_sample=sequence,
            )
        )

    primary_result = next(res for res in variant_results if res["label"] == DEPTH_MODE)
    comparison_results = [res for res in variant_results if res is not primary_result]

    tracking_summary = primary_result["tracking_summary"]
    ba_result = primary_result["ba_result"]
    ba_result_no_imu = primary_result["ba_result_no_imu"]
    primary_initial_label = f"Initial PnP ({primary_result['label']})"
    primary_optimized_label = f"Optimized BA ({primary_result['label']})"
    optimized_translation_norms = primary_result["optimized_translation_norms"]
    optimized_rotation_norms_deg = primary_result["optimized_rotation_norms_deg"]
    optimized_translation_norms_no_imu = primary_result["optimized_translation_norms_no_imu"]
    optimized_rotation_norms_deg_no_imu = primary_result["optimized_rotation_norms_deg_no_imu"]
    rms_before = primary_result["rms_before"]
    rms_after = primary_result["rms_after"]
    track_history = primary_result["track_history"]
    tracks = primary_result["tracks"]
    sequence_results = primary_result["sequence_results"]

    save_results(tracking_summary, ba_result, sequence)

    for result in variant_results:
        label = result["label"]
        result_ba = result["ba_result"]
        print(
            f"{'Bundle adjustment summary (with IMU)' if USE_IMU_FACTORS else 'Bundle adjustment summary'} "
            f"[{label}]"
        )
        print(f"  Frames optimised: {result_ba['frames_for_ba']}")
        print(f"  Landmarks optimised: {len(result_ba['landmark_original_indices'])}")
        print(
            f"  Reprojection RMS: {result['rms_before']:.3f} px -> {result['rms_after']:.3f} px"
        )
        # print(
        #     "  Translation error norms (m):",
        #     np.array2string(np.asarray(result["optimized_translation_norms"]), precision=3),
        # )
        # print(
        #     "  Rotation error norms (deg):",
        #     np.array2string(np.asarray(result["optimized_rotation_norms_deg"]), precision=2),
        # )
        print(f"Stereo Factors: {sum(result_ba['stereo_counts'].values())}")
        print(f"Mono Factors: {sum(result_ba['mono_counts'].values())}")
        dist_pf, pct_pf, pct_cum = distance_percentage_series(result["pose_errors"])
        if dist_pf.size:
            # print("  Translation error as % of distance:")
            print("    Distances (m):", np.array2string(dist_pf, precision=2))
            # print("    Per-frame (%):", np.array2string(pct_pf, precision=2))
            # print("    Cumulative mean (%):", np.array2string(pct_cum, precision=2))

        total_distance = result["pose_errors"][sorted(result["pose_errors"])[-1]]["ground_truth_distance"]
        translation_errors = np.asarray([result["pose_errors"][idx]["translation_norm"] for idx in sorted(result["pose_errors"])])
        print(f"Total distance travelled ({result['label']}): {np.sum(translation_errors):.2f} m")
        print(f"RMSE ({result['label']}): {np.sqrt(np.mean(translation_errors ** 2)):.2f} m")
        print(f"% RMSE per meter travelled ({result['label']}): {(np.sqrt(np.mean(translation_errors ** 2)) / total_distance) * 100:.2f}")

        if result["ba_result_no_imu"] is not None:
            result_no_imu = result["ba_result_no_imu"]
            rms_before_no_imu = float(np.sqrt(np.mean(result_no_imu["reprojection_before"] ** 2)))
            rms_after_no_imu = float(np.sqrt(np.mean(result_no_imu["reprojection_after"] ** 2)))
            print("Bundle adjustment summary (vision only)")
            print(f"  Frames optimised: {result_no_imu['frames_for_ba']}")
            print(f"  Landmarks optimised: {len(result_no_imu['landmark_original_indices'])}")
            print(f"  Reprojection RMS: {rms_before_no_imu:.3f} px -> {rms_after_no_imu:.3f} px")
            # print(
            #     "  Translation error norms (m):",
            #     np.array2string(np.asarray(result["optimized_translation_norms_no_imu"]), precision=3),
            # )
            # print(
            #     "  Rotation error norms (deg):",
            #     np.array2string(np.asarray(result["optimized_rotation_norms_deg_no_imu"]), precision=2),
            # )
            print(f"Stereo Factors: {sum(result_no_imu['stereo_counts'].values())}")
            print(f"Mono Factors: {sum(result_no_imu['mono_counts'].values())}")
            dist_no_imu, pct_no_imu, pct_no_imu_cum = distance_percentage_series(result["pose_errors_no_imu"])
            # if dist_no_imu.size:
            #     print("  Translation error as % of distance (vision only):")
            #     print("    Distances (m):", np.array2string(dist_no_imu, precision=2))
            #     print("    Per-frame (%):", np.array2string(pct_no_imu, precision=2))
            #     print("    Cumulative mean (%):", np.array2string(pct_no_imu_cum, precision=2))
                
            total_distance = result["pose_errors_no_imu"][sorted(result["pose_errors_no_imu"])[-1]]["ground_truth_distance"]
            translation_errors = np.asarray([result["pose_errors_no_imu"][idx]["translation_norm"] for idx in sorted(result["pose_errors_no_imu"])])
            print(f"Total distance travelled (vision only): {np.sum(translation_errors):.2f} m")
            print(f"RMSE (vision only): {np.sqrt(np.mean(translation_errors ** 2)):.2f} m")
            print(f"% RMSE per meter travelled (vision only): {(np.sqrt(np.mean(translation_errors ** 2)) / total_distance) * 100:.2f}")

    # plot_feature_tracks(rectified_frames, track_history, tracks)
    plot_track_length_histogram(tracks)
    # plot_match_debug(rectified_frames, sequence_results)

    additional_pose_dicts = None
    additional_initial_pose_dicts = None
    if comparison_results:
        additional_pose_entries = {
            f"Optimized BA ({res['label']})": res["optimized_pose_dict"] for res in comparison_results
            if res.get("optimized_pose_dict")
        }
        if additional_pose_entries:
            additional_pose_dicts = additional_pose_entries
        additional_initial_entries = {
            f"Initial PnP ({res['label']})": res["initial_pose_dict"] for res in comparison_results
            if res.get("initial_pose_dict")
        }
        if additional_initial_entries:
            additional_initial_pose_dicts = additional_initial_entries
    plot_pose_trajectories(
        ba_result["initial_pose_dict"],
        ba_result["optimized_pose_dict"],
        sequence,
        landmark_stride=5,
        inertial_poses=inertial_poses,
        optimized_pose_dict_no_imu=ba_result_no_imu["optimized_pose_dict"] if ba_result_no_imu else None,
        initial_label=primary_initial_label,
        optimized_label=primary_optimized_label,
        additional_pose_dicts=additional_pose_dicts,
        additional_initial_pose_dicts=additional_initial_pose_dicts,
    )
    plot_pose_error_norms_per_frame(
        ba_result["initial_pose_dict"],
        ba_result["optimized_pose_dict"],
        sequence,
        comparison_pose_dict=ba_result_no_imu["optimized_pose_dict"] if ba_result_no_imu else None,
    )
    plot_reprojection_error_histograms(
        ba_result["reprojection_before"],
        ba_result["reprojection_after"],
    )
    percentage_plot_series: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    err_xyzs = []
    for result in variant_results:
        dist_pf, pct_pf, pct_cum = distance_percentage_series(result["pose_errors"])
        if dist_pf.size:
            suffix = " (IMU)" if USE_IMU_FACTORS else ""
            percentage_plot_series.append((f"{result['label']}{suffix}", dist_pf, pct_pf, pct_cum))
        pose_errors_no_imu = result.get("pose_errors_no_imu")
        if pose_errors_no_imu:
            dist_no, pct_no, pct_no_cum = distance_percentage_series(pose_errors_no_imu)
            if dist_no.size:
                percentage_plot_series.append((f"{result['label']} (no IMU)", dist_no, pct_no, pct_no_cum))

        # total_distance = pose_errors_no_imu[sorted(pose_errors_no_imu)[-1]]["ground_truth_distance"]
        # translation_errors = np.asarray([pose_errors_no_imu[idx]["translation_norm"] for idx in sorted(pose_errors_no_imu)])
        # print(f"Total distance travelled ({result['label']}): {np.sum(translation_errors):.2f} m")
        # print(f"RMSE ({result['label']}): {np.sqrt(np.mean(translation_errors ** 2)):.2f} m")
        # print(f"RMSE per meter travelled ({result['label']}): {(np.sqrt(np.mean(translation_errors ** 2)) / total_distance):.3f}")

        # Pass the ground truth trajectory directly to the evaluation function.
        est_traj_xyz = np.asarray([result["optimized_pose_dict"][idx].translation() for idx in sorted(result["optimized_pose_dict"])])
        est_traj_quat = [result["optimized_pose_dict"][idx].rotation().toQuaternion() for idx in sorted(result["optimized_pose_dict"])]
        est_traj_qxqyqzqw = np.asarray([[quat.w(), quat.x(), quat.y(), quat.z()] for quat in est_traj_quat])
        est_traj = np.concatenate([est_traj_xyz, est_traj_qxqyqzqw], axis=1)

        gt_traj_xyz = np.asarray([sequence.world_poses[idx].translation() for idx in range(len(sequence.world_poses))])
        gt_traj_quat = [sequence.world_poses[idx].rotation().toQuaternion() for idx in range(len(sequence.world_poses))]
        gt_traj_qxqyqzqw = np.asarray([[quat.w(), quat.x(), quat.y(), quat.z()] for quat in gt_traj_quat])
        gt_traj = np.concatenate([gt_traj_xyz, gt_traj_qxqyqzqw], axis=1)

        ta_results = ta.evaluate_traj(est_traj,
                                gt_traj = gt_traj,
                                enforce_length = True,
                                plot = True,
                                plot_out_path = f"ta_results_{result['label']}.png",
                                do_scale = False,
                                do_align = True)
        for key, value in ta_results.items():
            print(f"{key}: {value}")
        
        ate = ta_results["ate"]
        est_traj_aligned = ta_results["est_traj"]
        est_traj_aligned_xyz = est_traj_aligned[:, :3]
        gt_traj = ta_results["gt_traj"]
        gt_traj_xyz = gt_traj[:, :3]
        gt_traj_xyz_diff = np.diff(gt_traj_xyz, axis=0)
        gt_traj_xyz_diff_norm = np.linalg.norm(gt_traj_xyz_diff, axis=1)
        gt_traj_total_distance = np.sum(gt_traj_xyz_diff_norm)

        err_xyz = np.linalg.norm(est_traj_aligned_xyz - gt_traj_xyz, axis=1)
        err_xyzs.append((result['label'], err_xyz))

        print(f"Result: {result['label']}")
        print(f"GT total distance: {gt_traj_total_distance:.2f} m")
        print(f"ATE per meter travelled: {(ate / gt_traj_total_distance) * 100:.2f}%")

        # NEXT ADD/REMOVE IMU NOISE
        if result['label'] == 'sgbm':
            rr.init("local_vo_bundle_adjustment")
            rr.spawn()

            est_poses = []
            gt_poses = []

            for i in range(len(est_traj_aligned)):
                rr.set_time("frame", sequence=i)

                est_pose = gtsam.Pose3(
                    gtsam.Rot3.Quaternion(w=est_traj_aligned[i, 3], x=est_traj_aligned[i, 4], y=est_traj_aligned[i, 5], z=est_traj_aligned[i, 6]),
                    gtsam.Point3(est_traj_aligned[i, 0:3])
                )
                est_poses.append(est_pose)
                rr_log_pose("est_pose", est_pose, depth_variants[1][1][i] if len(depth_variants) > 1 else depth_variants[0][1][i])

                gt_pose = gtsam.Pose3(
                    gtsam.Rot3.Quaternion(w=gt_traj[i, 6], x=gt_traj[i, 3], y=gt_traj[i, 4], z=gt_traj[i, 5]),
                    gtsam.Point3(gt_traj[i, 0:3])
                )
                gt_poses.append(gt_pose)
                # rr_log_pose("gt_pose", gt_pose, rectified_frames[0][1][i])

                rr_log_trajectory("est_trajectory", est_poses, color=(0, 0, 255), radii=0.006)
                rr_log_trajectory("gt_trajectory", gt_poses, color=(0, 255, 0), radii=0.006)

    plot_translation_error_percentages(percentage_plot_series)

    for label, err_xyz in err_xyzs:
        plt.plot(range(len(err_xyz)), err_xyz, label=label)
    plt.legend()
    plt.show()


    all_tracking_results.append(sequence_results)
    all_ba_results.append(ba_result)

# %%
