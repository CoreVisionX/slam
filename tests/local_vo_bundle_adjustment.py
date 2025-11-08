# %%
import os
import sys
from pathlib import Path
from typing import Any
import cv2
import gtsam
from gtsam.symbol_shorthand import L, X, V, B
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
SEQUENCE_LENGTH = 16
ENVIRONMENT = "AbandonedFactory"
DIFFICULTY = "easy"
TRAJECTORY = "P002"
SAMPLING_MODE = "stride"
MIN_STRIDE = 1
MAX_STRIDE = 1
BASE_SEED = 0

USE_IMU_FACTORS = False
DEPTH_MODE = os.environ.get("LOCAL_VO_DEPTH_MODE", "ground_truth").strip().lower()
if DEPTH_MODE not in {"sgbm", "ground_truth"}:
    raise ValueError("LOCAL_VO_DEPTH_MODE must be either 'sgbm' or 'ground_truth'.")
USE_GROUND_TRUTH_DEPTH = DEPTH_MODE == "ground_truth"

# Feature detection / tracking
MAX_FEATURE_COUNT = 1024
FAST_THRESHOLD = 25
FAST_NONMAX = True
FAST_BORDER = 12

LK_WIN_SIZE = (7, 7)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)
LK_MIN_EIG_THRESHOLD = 1e-4

MIN_MATCHES_FOR_PNP = 6
MAX_DEPTH = 40.0

# Bundle adjustment
MIN_OBSERVATIONS_PER_LANDMARK = 5
MIN_OBSERVATIONS_PER_FRAME = 5
PROJECTION_NOISE_PX = 1.0
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
)

IMU_GRAVITY_MAGNITUDE = 9.81
IMU_ACCEL_NOISE = 0.1
IMU_GYRO_NOISE = np.deg2rad(0.5)
IMU_INTEGRATION_NOISE = 5e-4
IMU_VELOCITY_PRIOR_SIGMA = 1.0
IMU_BIAS_PRIOR_SIGMAS = np.array(
    [
        0.01,
        0.01,
        0.01,
        np.deg2rad(0.5),
        np.deg2rad(0.5),
        np.deg2rad(0.5),
    ],
    dtype=float,
)
IMU_GRAVITY_VECTOR = np.array([0.0, -IMU_GRAVITY_MAGNITUDE, 0.0], dtype=float)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "klt_local_vo_bundle_adjustment.npz"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Depth source: {'ground_truth' if USE_GROUND_TRUTH_DEPTH else 'sgbm'}")

sgbm = SGBM(num_disparities=16 * 4, block_size=5, image_color="RGB")


# %%
# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
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
    depth_xyz = depth_map_to_xyz(depth_rect, rectified_frame.calibration.K_left_rect)
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
    params = gtsam.PreintegrationParams.MakeSharedU(IMU_GRAVITY_MAGNITUDE)
    params.n_gravity = IMU_GRAVITY_VECTOR
    params.setAccelerometerCovariance(np.eye(3) * (IMU_ACCEL_NOISE ** 2))
    params.setGyroscopeCovariance(np.eye(3) * (IMU_GYRO_NOISE ** 2))
    params.setIntegrationCovariance(np.eye(3) * (IMU_INTEGRATION_NOISE ** 2))
    return params


def _integrate_imu_batch(
    pim: gtsam.PreintegratedImuMeasurements,
    batch: test_utils.FrameImuMeasurements,
    next_timestamp: float,
) -> None:
    if batch is None:
        raise ValueError("IMU batch missing while attempting to integrate measurements.")

    current_time = batch.frame_timestamp
    if batch.timestamps.size == 0:
        dt = float(next_timestamp - current_time)
        if dt > 0.0:
            pim.integrateMeasurement(np.zeros(3), np.zeros(3), dt)
        return

    for sample_time, accel, gyro in zip(
        batch.timestamps,
        batch.linear_accelerations,
        batch.angular_velocities,
    ):
        dt = float(sample_time - current_time)
        if dt > 0.0:
            pim.integrateMeasurement(accel, gyro, dt)
        current_time = float(sample_time)

    dt = float(next_timestamp - current_time)
    if dt > 0.0:
        pim.integrateMeasurement(
            batch.linear_accelerations[-1],
            batch.angular_velocities[-1],
            dt,
        )


def preintegrate_between_frames(
    sequence: test_utils.FrameSequenceWithGroundTruth[StereoFrame],  # type: ignore[type-arg]
    start_idx: int,
    end_idx: int,
    params: gtsam.PreintegrationParams,
    bias: gtsam.imuBias.ConstantBias,
) -> gtsam.PreintegratedImuMeasurements:
    if sequence.full_imu_measurements is None or sequence.frame_timestamps is None:
        raise ValueError("IMU measurements unavailable for this sequence.")
    if end_idx <= start_idx:
        raise ValueError("end_idx must be greater than start_idx for IMU preintegration.")

    pim = gtsam.PreintegratedImuMeasurements(params, bias)
    imu_batches = sequence.full_imu_measurements
    timestamps = sequence.frame_timestamps
    frame_indices = sequence.frame_indices

    raw_start = frame_indices[start_idx]
    raw_end = frame_indices[end_idx]

    for raw_idx in range(raw_start, raw_end):
        if raw_idx >= len(imu_batches):
            break
        batch = imu_batches[raw_idx]
        if raw_idx + 1 >= len(timestamps):
            break
        next_timestamp = float(timestamps[raw_idx + 1])
        _integrate_imu_batch(pim, batch, next_timestamp)

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
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1.0
        velocities[idx] = (positions[idx] - positions[idx - 1]) / dt
    if sequence.length > 1:
        velocities[0] = velocities[1]
    return velocities


def integrate_inertial_trajectory(
    sequence: test_utils.FrameSequenceWithGroundTruth[StereoFrame],  # type: ignore[type-arg]
) -> list[gtsam.Pose3] | None:
    imu_batches = sequence.imu_measurements
    if not imu_batches:
        return None

    params = create_preintegration_params()
    bias = gtsam.imuBias.ConstantBias()
    velocities = estimate_sequence_velocities(sequence)
    initial_velocity = gtsam.Point3(*velocities[0]) if velocities.size else gtsam.Point3(0.0, 0.0, 0.0)

    nav_state = gtsam.NavState(sequence.world_poses[0], initial_velocity)
    inertial_poses: list[gtsam.Pose3] = [nav_state.pose()]

    for idx in range(1, sequence.length):
        pim = preintegrate_between_frames(sequence, idx - 1, idx, params, bias)
        nav_state = pim.predict(nav_state, bias)
        inertial_poses.append(nav_state.pose())

    return inertial_poses


def track_with_klt(
    gray_sequence: list[np.ndarray],
    anchor_keypoints: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    num_features = anchor_keypoints.shape[0]
    history: list[np.ndarray] = [anchor_keypoints.copy()]
    active_indices: list[np.ndarray] = [np.arange(num_features)]
    status_history: list[np.ndarray] = [np.ones(num_features, dtype=bool)]
    error_history: list[np.ndarray] = [np.zeros(num_features, dtype=np.float32)]

    prev_gray = gray_sequence[0]
    prev_points = anchor_keypoints.reshape(-1, 1, 2)
    prev_indices = np.arange(num_features, dtype=int)

    lk_params = dict(
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA,
        minEigThreshold=LK_MIN_EIG_THRESHOLD,
    )

    for gray in gray_sequence[1:]:
        if prev_points.size == 0:
            history.append(np.full((num_features, 2), np.nan, dtype=np.float32))
            active_indices.append(np.array([], dtype=int))
            status_history.append(np.zeros(num_features, dtype=bool))
            error_history.append(np.full(num_features, np.nan, dtype=np.float32))
            prev_gray = gray
            continue

        next_points, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
        status = status.reshape(-1).astype(bool)
        valid_indices = prev_indices[status]

        coords = np.full((num_features, 2), np.nan, dtype=np.float32)
        coords[valid_indices] = next_points[status].reshape(-1, 2)
        history.append(coords)

        mask = np.zeros(num_features, dtype=bool)
        mask[valid_indices] = True
        status_history.append(mask)
        active_indices.append(valid_indices)

        full_errors = np.full(num_features, np.nan, dtype=np.float32)
        if errors is not None:
            errors = errors.reshape(-1)
            full_errors[valid_indices] = errors[status]
        error_history.append(full_errors)

        prev_gray = gray
        prev_points = next_points[status].reshape(-1, 1, 2)
        prev_indices = valid_indices

    return history, active_indices, status_history, error_history


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
    track_history: list[np.ndarray],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    sequence_length = sequence_sample.length
    translation_errors = np.full((sequence_length, 3), np.nan, dtype=np.float64)
    rotation_errors = np.full((sequence_length, 3), np.nan, dtype=np.float64)
    translation_norms = np.full(sequence_length, np.nan, dtype=np.float64)
    rotation_norms_rad = np.full(sequence_length, np.nan, dtype=np.float64)
    inlier_counts = np.zeros(sequence_length, dtype=int)
    active_track_counts = np.array([np.sum(~np.isnan(points[:, 0])) for points in track_history], dtype=int)

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
    anchor_frame: FeatureFrame,
    rectified_frames: list[RectifiedStereoFrame],
    track_history: list[np.ndarray],
    active_indices: list[np.ndarray],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for frame_idx in range(1, len(rectified_frames)):
        tracked_indices = active_indices[frame_idx]
        if tracked_indices.size == 0:
            results.append(
                {
                    "frame_index": frame_idx,
                    "frame_id": sequence_sample.frame_ids[frame_idx],
                    "status": "no_tracks",
                    "active_track_count": 0,
                }
            )
            continue

        keypoints_second = track_history[frame_idx][tracked_indices]
        valid_mask = ~np.isnan(keypoints_second[:, 0])
        keypoints_second = keypoints_second[valid_mask]
        tracked_indices = tracked_indices[valid_mask]

        if keypoints_second.shape[0] < MIN_MATCHES_FOR_PNP:
            results.append(
                {
                    "frame_index": frame_idx,
                    "frame_id": sequence_sample.frame_ids[frame_idx],
                    "status": "insufficient_tracks",
                    "active_track_count": keypoints_second.shape[0],
                }
            )
            continue

        second_frame = make_feature_frame_for_view(rectified_frames[frame_idx], keypoints_second)
        matches = np.stack(
            [tracked_indices, np.arange(keypoints_second.shape[0], dtype=int)],
            axis=1,
        )
        matched_pair = MatchedFramePair(
            first=anchor_frame,
            second=second_frame,
            matches=matches,
        )

        try:
            estimated_pose, inlier_pair = solve_pnp(matched_pair)
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "frame_index": frame_idx,
                    "frame_id": sequence_sample.frame_ids[frame_idx],
                    "status": "pnp_failed",
                    "active_track_count": keypoints_second.shape[0],
                    "error": str(exc),
                }
            )
            continue

        ground_truth_pose = sequence_sample.relative_pose(0, frame_idx)
        pose_error = ground_truth_pose.between(estimated_pose)

        results.append(
            {
                "frame_index": frame_idx,
                "frame_id": sequence_sample.frame_ids[frame_idx],
                "status": "success",
                "active_track_count": keypoints_second.shape[0],
                "matches_before_filter": matches.shape[0],
                "matches_after_filter": inlier_pair.matches.shape[0],
                "estimated_pose": estimated_pose,
                "ground_truth_pose": ground_truth_pose,
                "pose_error": pose_error,
                "matched_pair": inlier_pair,
            }
        )

    return results


def compute_pose_initializations(
    sequence_results: list[dict[str, Any]],
    sequence_length: int,
) -> list[gtsam.Pose3]:
    initializations: list[gtsam.Pose3] = [gtsam.Pose3.Identity()]
    result_by_frame = {result["frame_index"]: result for result in sequence_results if result["status"] == "success"}
    current_pose = gtsam.Pose3.Identity()

    for frame_idx in range(1, sequence_length):
        if frame_idx in result_by_frame:
            current_pose = result_by_frame[frame_idx]["estimated_pose"]
        initializations.append(current_pose)

    return initializations


def select_frames_for_ba(
    track_history: list[np.ndarray],
) -> list[int]:
    selected = []
    for idx, points in enumerate(track_history):
        visible = np.sum(~np.isnan(points[:, 0]))
        if visible >= MIN_OBSERVATIONS_PER_FRAME or idx == 0:
            selected.append(idx)
    if 0 not in selected:
        selected.insert(0, 0)
    return selected


def build_calibration(frame: RectifiedStereoFrame) -> gtsam.Cal3_S2:
    K = frame.calibration.K_left_rect
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    skew = float(K[0, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return gtsam.Cal3_S2(fx, fy, skew, cx, cy)


def build_stereo_calibration(frame: RectifiedStereoFrame) -> gtsam.Cal3_S2Stereo:
    calib = frame.calibration
    K = calib.K_left_rect
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
    track_history: list[np.ndarray],
    attributes: dict[str, Any],
    sequence_results: list[dict[str, Any]],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
    use_imu: bool = True,
) -> dict[str, Any]:
    frames_for_ba = select_frames_for_ba(track_history)
    calibration = build_calibration(rectified_frames[0])
    stereo_calibration = build_stereo_calibration(rectified_frames[0])

    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, PROJECTION_NOISE_PX)
    stereo_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([PROJECTION_NOISE_PX, PROJECTION_NOISE_PX, PROJECTION_NOISE_PX], dtype=float)
    )
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(POSE_PRIOR_SIGMAS)

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    pose_initials = compute_pose_initializations(sequence_results, len(rectified_frames))
    sequence_velocities = estimate_sequence_velocities(sequence_sample)

    for frame_idx in frames_for_ba:
        initial_pose = clone_pose(pose_initials[frame_idx])
        values.insert(X(frame_idx), initial_pose)

    imu_available = use_imu and sequence_sample.imu_measurements is not None
    imu_params = create_preintegration_params() if imu_available else None
    imu_bias_key = None
    imu_bias_initial = gtsam.imuBias.ConstantBias()

    if imu_available:
        velocity_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, IMU_VELOCITY_PRIOR_SIGMA)
        bias_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(IMU_BIAS_PRIOR_SIGMAS)
        imu_bias_key = B(0)
        if not values.exists(imu_bias_key):
            values.insert(imu_bias_key, imu_bias_initial)

        for frame_idx in frames_for_ba:
            if not values.exists(V(frame_idx)):
                vel = (
                    sequence_velocities[frame_idx]
                    if frame_idx < sequence_velocities.shape[0]
                    else np.zeros(3)
                )
                values.insert(V(frame_idx), vel)

        first_velocity = (
            sequence_velocities[frames_for_ba[0]]
            if frames_for_ba[0] < sequence_velocities.shape[0]
            else np.zeros(3)
        )
        graph.add(gtsam.PriorFactorVector(V(frames_for_ba[0]), first_velocity, velocity_prior_noise))
        graph.add(gtsam.PriorFactorConstantBias(imu_bias_key, imu_bias_initial, bias_prior_noise))

    graph.add(gtsam.PriorFactorPose3(X(0), gtsam.Pose3.Identity(), prior_noise))

    if imu_available and imu_params is not None and imu_bias_key is not None:
        for prev_frame, next_frame in zip(frames_for_ba[:-1], frames_for_ba[1:]):
            if next_frame <= prev_frame:
                continue
            pim = preintegrate_between_frames(
                sequence_sample,
                prev_frame,
                next_frame,
                imu_params,
                imu_bias_initial,
            )
            graph.add(
                gtsam.ImuFactor(
                    X(prev_frame),
                    V(prev_frame),
                    X(next_frame),
                    V(next_frame),
                    imu_bias_key,
                    pim,
                )
            )

    landmark_positions = attributes["keypoints_3d"]
    anchor_depths = attributes["keypoints_depth"]
    anchor_measurements = attributes["keypoints"]
    landmark_keys: list[int] = []
    landmark_index_lookup: list[int] = []
    observations: list[tuple[int, int, np.ndarray]] = []
    stereo_counts = {frame_idx: 0 for frame_idx in frames_for_ba}
    mono_counts = {frame_idx: 0 for frame_idx in frames_for_ba}

    inlier_lookup: dict[int, set[int]] = {}
    for result in sequence_results:
        if result.get("status") != "success":
            continue
        frame_idx = result["frame_index"]
        matches = result["matched_pair"].matches
        inlier_lookup.setdefault(frame_idx, set()).update(matches[:, 0].tolist())

    for landmark_idx, point in enumerate(landmark_positions):
        observation_frames: list[tuple[int, np.ndarray]] = []
        for frame_idx in frames_for_ba:
            measurement = track_history[frame_idx][landmark_idx]
            if np.isnan(measurement[0]):
                continue
            if frame_idx != 0 and frame_idx in inlier_lookup:
                if landmark_idx not in inlier_lookup[frame_idx]:
                    continue
            observation_frames.append((frame_idx, measurement))

        if len(observation_frames) < MIN_OBSERVATIONS_PER_LANDMARK:
            continue

        landmark_key = len(landmark_keys)
        landmark_keys.append(landmark_key)
        landmark_index_lookup.append(landmark_idx)

        values.insert(L(landmark_key), point3_like_to_numpy(point))

        for frame_idx, measurement in observation_frames:
            if frame_idx == 0:
                depth_value = float(anchor_depths[landmark_idx])
                if not np.isfinite(depth_value) or depth_value <= 0.0:
                    continue
                fx = stereo_calibration.fx()
                baseline = stereo_calibration.baseline()
                disparity = (fx * baseline) / depth_value
                stereo_measurement = gtsam.StereoPoint2(
                    float(anchor_measurements[landmark_idx][0]),
                    float(anchor_measurements[landmark_idx][0] - disparity),
                    float(anchor_measurements[landmark_idx][1]),
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
                        gtsam.Point2(float(measurement[0]), float(measurement[1])),
                        measurement_noise,
                        X(frame_idx),
                        L(landmark_key),
                        calibration,
                    )
                )
                mono_counts[frame_idx] += 1
            observations.append((frame_idx, landmark_key, measurement))

    if not observations:
        raise RuntimeError("No landmark observations available for bundle adjustment.")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values)
    optimized_values = optimizer.optimize()

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
    initial_landmarks = [point3_like_to_numpy(landmark_positions[idx]) for idx in landmark_index_lookup]

    observation_matrix = [(frame_idx, landmark_idx, measurement) for frame_idx, landmark_idx, measurement in observations]

    reprojection_before = compute_reprojection_errors(initial_pose_dict, initial_landmarks, observation_matrix, calibration)
    reprojection_after = compute_reprojection_errors(optimized_pose_dict, optimized_landmarks, observation_matrix, calibration)

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
    }


def compute_pose_errors_against_ground_truth(
    optimized_pose_dict: dict[int, gtsam.Pose3],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
) -> dict[int, dict[str, np.ndarray]]:
    errors: dict[int, dict[str, np.ndarray]] = {}
    for frame_idx, pose in optimized_pose_dict.items():
        gt_pose = sequence_sample.relative_pose(0, frame_idx)
        error = gt_pose.between(pose)
        trans, rot = pose_error_components(error)
        errors[frame_idx] = {
            "translation": trans,
            "rotation": rot,
            "translation_norm": np.linalg.norm(trans),
            "rotation_norm_rad": np.linalg.norm(rot),
        }
    return errors


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
    track_history: list[np.ndarray],
) -> None:
    anchor_image = rectified_frames[0].left_rect
    anchor_points = track_history[0]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].set_title("Anchor frame with FAST keypoints")
    axes[0].imshow(anchor_image)
    axes[0].scatter(anchor_points[:, 0], anchor_points[:, 1], s=10, c="lime", linewidths=0.5)
    axes[0].axis("off")

    last_frame_image = rectified_frames[-1].left_rect
    axes[1].set_title("Tracked KLT trajectories")
    axes[1].imshow(last_frame_image)
    cmap = plt.cm.get_cmap("viridis", max(1, anchor_points.shape[0]))
    for kp_idx in range(anchor_points.shape[0]):
        path = np.array([frame[kp_idx] for frame in track_history])
        valid = ~np.isnan(path[:, 0])
        if np.count_nonzero(valid) < 2:
            continue
        axes[1].plot(path[valid, 0], path[valid, 1], color=cmap(kp_idx % cmap.N), linewidth=1.0)
        axes[1].scatter(path[valid][-1, 0], path[valid][-1, 1], s=8, color=cmap(kp_idx % cmap.N))
    axes[1].axis("off")

    frame_indices = list(range(len(track_history)))
    track_counts = [np.sum(~np.isnan(frame[:, 0])) for frame in track_history]

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
    pair = FramePair(first=rectified_frames[0], second=rectified_frames[frame_idx])

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
            to_numpy_vec3((base_initial.inverse() * initial_pose_dict[f]).translation())
            for f in frames
        ],
        dtype=np.float64,
    )
    optimized_xyz = np.array(
        [
            to_numpy_vec3((base_optimized.inverse() * optimized_pose_dict[f]).translation())
            for f in frames
        ],
        dtype=np.float64,
    )
    ground_truth_xyz = np.array(
        [to_numpy_vec3(sequence_sample.relative_pose(0, f).translation()) for f in frames],
        dtype=np.float64,
    )
    optimized_xyz_no_imu: np.ndarray | None = None
    if optimized_pose_dict_no_imu is not None:
        optimized_xyz_no_imu = np.array(
            [
                to_numpy_vec3(
                    (base_optimized.inverse() * optimized_pose_dict_no_imu[f]).translation()
                )
                for f in frames
            ],
            dtype=np.float64,
        )

    inertial_xyz: np.ndarray | None = None
    if inertial_poses:
        try:
            inertial_points: list[np.ndarray] = []
            for idx, frame in enumerate(frames):
                if frame >= len(inertial_poses):
                    break
                imu_pose = inertial_poses[frame]
                gt_pose_abs = sequence_sample.world_poses[frame]
                gt_rel_pose = sequence_sample.relative_pose(0, frame)
                relative_pose = gt_pose_abs.inverse() * imu_pose
                rel_translation = to_numpy_vec3(relative_pose.translation())
                rot = gt_rel_pose.rotation().matrix()
                rel_world = rot @ rel_translation
                inertial_points.append(ground_truth_xyz[idx] + rel_world)
            if inertial_points:
                inertial_xyz = np.asarray(inertial_points, dtype=np.float64)
        except IndexError:
            print("IMU trajectory shorter than pose set; skipping inertial overlay.")
            inertial_xyz = None

    plt.figure(figsize=(10, 6))
    plt.title("Trajectory comparison (XY plane)")
    plt.plot(initial_xyz[:, 0], initial_xyz[:, 1], "o--", label=initial_label)
    plt.plot(optimized_xyz[:, 0], optimized_xyz[:, 1], "o-", label=optimized_label)
    plt.plot(ground_truth_xyz[:, 0], ground_truth_xyz[:, 1], "x-", label="Ground truth")
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
                relative_pose = base_pose.inverse() * pose
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
                relative_pose = base_pose.inverse() * pose_dict[frame]
                additional_points.append(to_numpy_vec3(relative_pose.translation()))
            if not additional_points:
                continue
            additional_xyz = np.asarray(additional_points, dtype=np.float64)
            plt.plot(additional_xyz[:, 0], additional_xyz[:, 1], "o-", label=label)
    if inertial_xyz is not None:
        plt.plot(
            inertial_xyz[:, 0],
            inertial_xyz[:, 1],
            "s",
            label="IMU-only (per-frame)",
        )

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
    axes[0].plot(shared_frames, translation_before, "o--", label="Before BA")
    axes[0].plot(shared_frames, translation_after, "o-", label="After BA (IMU)")
    if comparison_errors is not None:
        translation_comp = [comparison_errors[idx]["translation_norm"] for idx in shared_frames]
        axes[0].plot(shared_frames, translation_comp, "o-", label="After BA (no IMU)")
    axes[0].set_xlabel("Frame index")
    axes[0].set_ylabel("Translation error (m)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_title("Rotation error norms")
    axes[1].plot(shared_frames, rotation_before_deg, "o--", label="Before BA")
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


def run_depth_variant(
    depth_label: str,
    rectified_frames: list[RectifiedStereoFrame],
    depth_frames: list[StereoDepthFrame],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
) -> dict[str, Any]:
    gray_sequence = [cv2.cvtColor(frame.left_rect, cv2.COLOR_RGB2GRAY) for frame in rectified_frames]
    anchor_keypoints = detect_fast_keypoints(
        gray_sequence[0],
        max_features=MAX_FEATURE_COUNT,
        threshold=FAST_THRESHOLD,
        nonmax=FAST_NONMAX,
        border=FAST_BORDER,
    )

    attributes = extract_keypoint_attributes(depth_frames[0], anchor_keypoints, max_depth=MAX_DEPTH)
    filtered_keypoints = attributes["keypoints"]
    if filtered_keypoints.shape[0] < MIN_MATCHES_FOR_PNP:
        raise RuntimeError(
            f"[{depth_label}] Only {filtered_keypoints.shape[0]} keypoints with valid depth after filtering; "
            f"consider relaxing FAST/border/max-depth settings."
        )

    anchor_frame = build_feature_frame(depth_frames[0], attributes)
    track_history, active_indices, _, _ = track_with_klt(gray_sequence, filtered_keypoints.astype(np.float32))

    sequence_results = estimate_sequence_poses(
        anchor_frame=anchor_frame,
        rectified_frames=rectified_frames,
        track_history=track_history,
        active_indices=active_indices,
        sequence_sample=sequence_sample,
    )

    tracking_summary = summarise_tracking_results(sequence_sample, track_history, sequence_results)

    ba_result = run_bundle_adjustment(
        rectified_frames=rectified_frames,
        track_history=track_history,
        attributes=attributes,
        sequence_results=sequence_results,
        sequence_sample=sequence_sample,
        use_imu=USE_IMU_FACTORS,
    )

    ba_result_no_imu = None
    if USE_IMU_FACTORS:
        ba_result_no_imu = run_bundle_adjustment(
            rectified_frames=rectified_frames,
            track_history=track_history,
            attributes=attributes,
            sequence_results=sequence_results,
            sequence_sample=sequence_sample,
            use_imu=False,
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
    )
    inertial_poses = integrate_inertial_trajectory(sequence)

    rectified_frames = [frame.rectify() for frame in sequence.frames]
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
        print(
            "  Translation error norms (m):",
            np.array2string(np.asarray(result["optimized_translation_norms"]), precision=3),
        )
        print(
            "  Rotation error norms (deg):",
            np.array2string(np.asarray(result["optimized_rotation_norms_deg"]), precision=2),
        )
        print(f"Stereo Factors: {sum(result_ba['stereo_counts'].values())}")
        print(f"Mono Factors: {sum(result_ba['mono_counts'].values())}")

        if result["ba_result_no_imu"] is not None:
            result_no_imu = result["ba_result_no_imu"]
            rms_before_no_imu = float(np.sqrt(np.mean(result_no_imu["reprojection_before"] ** 2)))
            rms_after_no_imu = float(np.sqrt(np.mean(result_no_imu["reprojection_after"] ** 2)))
            print("Bundle adjustment summary (vision only)")
            print(f"  Frames optimised: {result_no_imu['frames_for_ba']}")
            print(f"  Landmarks optimised: {len(result_no_imu['landmark_original_indices'])}")
            print(f"  Reprojection RMS: {rms_before_no_imu:.3f} px -> {rms_after_no_imu:.3f} px")
            print(
                "  Translation error norms (m):",
                np.array2string(np.asarray(result["optimized_translation_norms_no_imu"]), precision=3),
            )
            print(
                "  Rotation error norms (deg):",
                np.array2string(np.asarray(result["optimized_rotation_norms_deg_no_imu"]), precision=2),
            )
            print(f"Stereo Factors: {sum(result_no_imu['stereo_counts'].values())}")
            print(f"Mono Factors: {sum(result_no_imu['mono_counts'].values())}")

    plot_feature_tracks(rectified_frames, track_history)
    plot_match_debug(rectified_frames, sequence_results)

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
        # inertial_poses=inertial_poses,
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

    all_tracking_results.append(sequence_results)
    all_ba_results.append(ba_result)

# %%
