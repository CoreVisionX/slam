# %%
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import gtsam
from gtsam.symbol_shorthand import L, X
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
    StereoDepthFrame,
)
from registration.utils import draw_matches, solve_pnp  # noqa: E402
import tests.test_utils as test_utils  # noqa: E402

# TODO: expressing disparity uncertainty for depth measurements in BA properly might help a ton

# %%
# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
NUM_SEQUENCE_SAMPLES = 1
SEQUENCE_LENGTH = 40
ENVIRONMENT = "AbandonedFactory"
DIFFICULTY = "easy"
TRAJECTORY = "P001"
SAMPLING_MODE = "stride"
MIN_STRIDE = 1
MAX_STRIDE = 1
BASE_SEED = 4

# Feature detection / tracking
MAX_FEATURE_COUNT = 1024
FAST_THRESHOLD = 25
FAST_NONMAX = True
FAST_BORDER = 12
FEATURE_REFRESH_RATIO = 0.90
NEW_FEATURE_MIN_DISTANCE_PX = 5.0
REPROJECTION_GATING_THRESHOLD_PX = 8.0
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)
LK_MIN_EIG_THRESHOLD = 1e-4

MIN_MATCHES_FOR_PNP = 30
MAX_DEPTH = 20.0

# Bundle adjustment
INLIERS_ONLY = False
USE_HUBER = True
MIN_OBSERVATIONS_PER_LANDMARK = 10
MIN_OBSERVATIONS_PER_FRAME = 15
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

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "klt_local_vo_bundle_adjustment.npz"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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


def filter_candidate_points(
    existing_points: np.ndarray,
    candidate_points: np.ndarray,
    min_distance: float,
) -> np.ndarray:
    if candidate_points.size == 0:
        return candidate_points
    if existing_points.size == 0:
        return candidate_points

    valid_mask = ~np.isnan(existing_points[:, 0])
    existing = existing_points[valid_mask]
    if existing.size == 0:
        return candidate_points

    filtered: list[np.ndarray] = []
    for point in candidate_points:
        deltas = existing - point
        distances = np.linalg.norm(deltas, axis=1)
        if np.all(distances >= min_distance):
            filtered.append(point)
    if not filtered:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(filtered, dtype=np.float32)


def augment_tracks_with_new_features(
    gray_sequence: list[np.ndarray],
    depth_frames: list[StereoDepthFrame],
    track_history: list[np.ndarray],
    attributes: dict[str, Any],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    num_frames = len(gray_sequence)
    if num_frames == 0:
        return track_history, attributes

    birth_frames = attributes.get("birth_frames")
    if birth_frames is None:
        birth_frames = np.zeros(attributes["keypoints"].shape[0], dtype=int)

    threshold = int(np.floor(FEATURE_REFRESH_RATIO * MAX_FEATURE_COUNT))
    lk_params = dict(
        winSize=LK_WIN_SIZE,
        maxLevel=LK_MAX_LEVEL,
        criteria=LK_CRITERIA,
        minEigThreshold=LK_MIN_EIG_THRESHOLD,
    )

    total_features = track_history[0].shape[0]
    birth_frames = np.asarray(birth_frames, dtype=int)

    for frame_idx in range(1, num_frames - 1):
        current_tracks = track_history[frame_idx]
        active_mask = ~np.isnan(current_tracks[:, 0])
        active_count = int(np.count_nonzero(active_mask))
        if active_count >= threshold:
            continue

        needed = MAX_FEATURE_COUNT - active_count
        if needed <= 0:
            continue

        candidate_points = detect_fast_keypoints(
            gray_sequence[frame_idx],
            max_features=MAX_FEATURE_COUNT * 2,
            threshold=FAST_THRESHOLD,
            nonmax=FAST_NONMAX,
            border=FAST_BORDER,
        )
        existing_points = current_tracks
        filtered_points = filter_candidate_points(
            existing_points,
            candidate_points,
            NEW_FEATURE_MIN_DISTANCE_PX,
        )
        if filtered_points.size == 0:
            continue

        additions = extract_keypoint_attributes(
            depth_frames[frame_idx],
            filtered_points,
            max_depth=MAX_DEPTH,
        )
        new_keypoints = additions["keypoints"]
        if new_keypoints.shape[0] == 0:
            continue

        keep_count = min(new_keypoints.shape[0], max(needed, 0))
        new_keypoints = new_keypoints[:keep_count]
        new_depths = additions["keypoints_depth"][:keep_count]
        new_points3d = additions["keypoints_3d"][:keep_count]
        new_colors = additions["keypoints_color"][:keep_count]

        num_new = new_keypoints.shape[0]
        if num_new == 0:
            continue

        # Extend track history with placeholder rows.
        for history_idx in range(num_frames):
            placeholder = np.full((num_new, 2), np.nan, dtype=np.float32)
            track_history[history_idx] = np.vstack([track_history[history_idx], placeholder])

        # Set anchor positions at the birth frame.
        track_history[frame_idx][-num_new:, :] = new_keypoints.astype(np.float32)

        prev_points = new_keypoints.reshape(-1, 1, 2).astype(np.float32)
        prev_gray = gray_sequence[frame_idx]
        active_indices = np.arange(num_new, dtype=int)

        for future_idx in range(frame_idx + 1, num_frames):
            coords = np.full((num_new, 2), np.nan, dtype=np.float32)
            if prev_points.size == 0:
                track_history[future_idx][-num_new:, :] = coords
                continue

            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray_sequence[future_idx],
                prev_points,
                None,
                **lk_params,
            )

            if next_points is not None and status is not None:
                status = status.reshape(-1).astype(bool)
                if np.any(status):
                    coords[active_indices[status]] = next_points[status].reshape(-1, 2)
                    prev_points = next_points[status].reshape(-1, 1, 2)
                    active_indices = active_indices[status]
                else:
                    prev_points = np.empty((0, 1, 2), dtype=np.float32)
                    active_indices = np.empty((0,), dtype=int)
            else:
                prev_points = np.empty((0, 1, 2), dtype=np.float32)
                active_indices = np.empty((0,), dtype=int)

            track_history[future_idx][-num_new:, :] = coords
            prev_gray = gray_sequence[future_idx]

        # Update attribute arrays and birth frames.
        attributes["keypoints"] = np.vstack([attributes["keypoints"], new_keypoints])
        attributes["keypoints_depth"] = np.concatenate([attributes["keypoints_depth"], new_depths])
        attributes["keypoints_3d"] = np.vstack([attributes["keypoints_3d"], new_points3d])
        attributes["keypoints_color"] = np.vstack([attributes["keypoints_color"], new_colors])
        birth_frames = np.concatenate([birth_frames, np.full(num_new, frame_idx, dtype=int)])
        total_features += num_new

    attributes["birth_frames"] = birth_frames
    return track_history, attributes


def compute_active_indices_from_history(track_history: list[np.ndarray]) -> list[np.ndarray]:
    active_indices: list[np.ndarray] = []
    for frame_points in track_history:
        indices = np.where(~np.isnan(frame_points[:, 0]))[0].astype(int)
        active_indices.append(indices)
    return active_indices


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
    rectified_frames: list[RectifiedStereoFrame],
    depth_frames: list[StereoDepthFrame],
    track_history: list[np.ndarray],
    active_indices: list[np.ndarray],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for frame_idx in range(1, len(rectified_frames)):
        prev_frame_idx = frame_idx - 1
        prev_coords_all = track_history[prev_frame_idx]
        curr_coords_all = track_history[frame_idx]

        shared_mask = (~np.isnan(prev_coords_all[:, 0])) & (~np.isnan(curr_coords_all[:, 0]))
        shared_indices = np.where(shared_mask)[0]

        if shared_indices.size == 0:
            results.append(
                {
                    "frame_index": frame_idx,
                    "frame_id": sequence_sample.frame_ids[frame_idx],
                    "status": "no_tracks",
                    "active_track_count": 0,
                }
            )
            continue

        prev_points_2d = prev_coords_all[shared_indices]
        curr_points_2d = curr_coords_all[shared_indices]

        if prev_points_2d.shape[0] < MIN_MATCHES_FOR_PNP:
            results.append(
                {
                    "frame_index": frame_idx,
                    "frame_id": sequence_sample.frame_ids[frame_idx],
                    "status": "insufficient_tracks",
                    "active_track_count": prev_points_2d.shape[0],
                }
            )
            continue

        depth_frame_prev = depth_frames[prev_frame_idx]
        rectified_curr = rectified_frames[frame_idx]
        depth_map = depth_frame_prev.left_depth
        xyz_map = depth_frame_prev.left_depth_xyz
        color_map = depth_frame_prev.left_rect

        valid_prev_points: list[np.ndarray] = []
        valid_prev_depths: list[float] = []
        valid_prev_xyz: list[np.ndarray] = []
        valid_prev_colors: list[np.ndarray] = []
        valid_curr_points: list[np.ndarray] = []
        valid_global_indices: list[int] = []

        h, w = depth_map.shape

        for shared_idx in shared_indices:
            prev_pt = prev_coords_all[shared_idx]
            curr_pt = curr_coords_all[shared_idx]
            c_prev = int(np.round(prev_pt[0]))
            r_prev = int(np.round(prev_pt[1]))
            if not (0 <= r_prev < h and 0 <= c_prev < w):
                continue
            depth_value = float(depth_map[r_prev, c_prev])
            if not np.isfinite(depth_value) or depth_value <= 0.0:
                continue
            xyz = xyz_map[r_prev, c_prev]
            if not np.all(np.isfinite(xyz)):
                continue
            valid_prev_points.append(prev_pt)
            valid_prev_depths.append(depth_value)
            valid_prev_xyz.append(xyz)
            if color_map is not None:
                valid_prev_colors.append(color_map[r_prev, c_prev])
            valid_curr_points.append(curr_pt)
            valid_global_indices.append(int(shared_idx))

        if len(valid_prev_points) < MIN_MATCHES_FOR_PNP:
            results.append(
                {
                    "frame_index": frame_idx,
                    "frame_id": sequence_sample.frame_ids[frame_idx],
                    "status": "insufficient_depth",
                    "active_track_count": len(valid_prev_points),
                }
            )
            continue

        prev_attributes: dict[str, Any] = {
            "keypoints": np.asarray(valid_prev_points, dtype=np.float32),
            "keypoints_depth": np.asarray(valid_prev_depths, dtype=np.float32),
            "keypoints_3d": np.asarray(valid_prev_xyz, dtype=np.float32),
            "keypoints_color": (
                np.asarray(valid_prev_colors, dtype=np.float32)
                if valid_prev_colors
                else np.zeros((len(valid_prev_points), 3), dtype=np.float32)
            ),
        }

        anchor_frame = build_feature_frame(depth_frame_prev, prev_attributes)
        target_frame = make_feature_frame_for_view(rectified_curr, np.asarray(valid_curr_points, dtype=np.float32))
        matches = np.stack([np.arange(len(valid_prev_points), dtype=int)] * 2, axis=1)
        matched_pair = MatchedFramePair(
            first=anchor_frame,
            second=target_frame,
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
                    "active_track_count": len(valid_prev_points),
                    "error": str(exc),
                }
            )
            continue

        ground_truth_pose = sequence_sample.relative_pose(frame_idx - 1, frame_idx)
        pose_error = ground_truth_pose.between(estimated_pose)

        inlier_globals = np.asarray(
            [valid_global_indices[idx] for idx in inlier_pair.matches[:, 0]], dtype=int
        )

        results.append(
            {
                "frame_index": frame_idx,
                "frame_id": sequence_sample.frame_ids[frame_idx],
                "status": "success",
                "active_track_count": len(valid_prev_points),
                "matches_before_filter": matches.shape[0],
                "matches_after_filter": inlier_pair.matches.shape[0],
                "estimated_pose": estimated_pose,
                "ground_truth_pose": ground_truth_pose,
                "pose_error": pose_error,
                "matched_pair": inlier_pair,
                "global_inlier_indices": inlier_globals,
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
            relative = result_by_frame[frame_idx]["estimated_pose"]
            current_pose = current_pose.compose(relative)
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
    depth_frames: list[StereoDepthFrame],
) -> dict[str, Any]:
    frames_for_ba = select_frames_for_ba(track_history)
    calibration = build_calibration(rectified_frames[0])
    stereo_calibration = build_stereo_calibration(rectified_frames[0])

    if USE_HUBER:
        measurement_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber(1.5),
            gtsam.noiseModel.Isotropic.Sigma(2, PROJECTION_NOISE_PX),
        )
    else:
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, PROJECTION_NOISE_PX)

    stereo_noise = gtsam.noiseModel.Diagonal.Sigmas(
        2.0 * np.array([PROJECTION_NOISE_PX, PROJECTION_NOISE_PX, PROJECTION_NOISE_PX], dtype=float)
    )
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(POSE_PRIOR_SIGMAS)

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    pose_initials = compute_pose_initializations(sequence_results, len(rectified_frames))

    for frame_idx in frames_for_ba:
        initial_pose = clone_pose(pose_initials[frame_idx])
        values.insert(X(frame_idx), initial_pose)

    graph.add(gtsam.PriorFactorPose3(X(0), gtsam.Pose3.Identity(), prior_noise))

    landmark_positions = attributes["keypoints_3d"]
    landmark_birth_frames = attributes.get(
        "birth_frames",
        np.zeros(len(landmark_positions), dtype=int),
    )
    landmark_birth_frames = np.asarray(landmark_birth_frames, dtype=int)
    landmark_depths = attributes.get("keypoints_depth", np.full(len(landmark_positions), np.nan))
    landmark_depths = np.asarray(landmark_depths, dtype=float)
    landmark_keys: list[int] = []
    landmark_index_lookup: list[int] = []
    observations: list[tuple[int, int, np.ndarray]] = []
    stereo_counts = {frame_idx: 0 for frame_idx in frames_for_ba}
    mono_counts = {frame_idx: 0 for frame_idx in frames_for_ba}
    raw_observation_counts = {frame_idx: 0 for frame_idx in frames_for_ba}
    gated_observation_counts = {frame_idx: 0 for frame_idx in frames_for_ba}
    stereo_attempt_counts = {frame_idx: 0 for frame_idx in frames_for_ba}
    stereo_success_counts = {frame_idx: 0 for frame_idx in frames_for_ba}
    residual_norms: dict[int, list[float]] = {frame_idx: [] for frame_idx in frames_for_ba}
    nan_landmark_indices: list[int] = []

    inlier_lookup: dict[int, set[int]] = {}
    for result in sequence_results:
        if result.get("status") != "success":
            continue
        frame_idx = result["frame_index"]
        global_inliers = result.get("global_inlier_indices")
        if global_inliers is not None:
            indices = np.asarray(global_inliers, dtype=int).tolist()
        else:
            matches = result["matched_pair"].matches
            indices = matches[:, 0].tolist()
        inlier_lookup.setdefault(frame_idx, set()).update(indices)

    initial_landmarks_world: list[np.ndarray] = []

    for landmark_idx, point in enumerate(landmark_positions):
        birth_frame = int(landmark_birth_frames[landmark_idx])
        birth_pose = pose_initials[birth_frame]
        point_local = point3_like_to_numpy(point)
        if not np.all(np.isfinite(point_local)):
            nan_landmark_indices.append(landmark_idx)
            continue
        world_point = birth_pose.transformFrom(gtsam.Point3(*point_local))
        world_point_np = point3_like_to_numpy(world_point)

        observation_frames: list[tuple[int, np.ndarray]] = []
        for frame_idx in frames_for_ba:
            measurement = track_history[frame_idx][landmark_idx]
            if np.isnan(measurement[0]):
                continue
            measurement_vec = np.asarray(measurement, dtype=np.float64)
            raw_observation_counts[frame_idx] += 1

            # inlier check
            if INLIERS_ONLY:
                if birth_frame == 0 and frame_idx != 0 and frame_idx in inlier_lookup:
                    if landmark_idx not in inlier_lookup[frame_idx]:
                        continue

            pose_initial = pose_initials[frame_idx]
            camera_initial = gtsam.PinholeCameraCal3_S2(pose_initial, calibration)
            try:
                predicted = camera_initial.project(world_point_np)
            except RuntimeError:
                continue
            if hasattr(predicted, "vector"):
                predicted_vec = predicted.vector()
            else:
                predicted_vec = np.asarray(predicted, dtype=np.float64)

            residual = predicted_vec - measurement_vec
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm > REPROJECTION_GATING_THRESHOLD_PX:
                continue
            observation_frames.append((frame_idx, measurement_vec))
            gated_observation_counts[frame_idx] += 1
            residual_norms[frame_idx].append(residual_norm)

        if len(observation_frames) < MIN_OBSERVATIONS_PER_LANDMARK:
            continue

        landmark_key = len(landmark_keys)
        landmark_keys.append(landmark_key)
        landmark_index_lookup.append(landmark_idx)

        values.insert(L(landmark_key), world_point)
        initial_landmarks_world.append(point3_like_to_numpy(world_point))

        fx = stereo_calibration.fx()
        baseline = stereo_calibration.baseline()

        for frame_idx, measurement in observation_frames:
            stereo_measurement = None
            depth_map = getattr(depth_frames[frame_idx], "left_depth", None)
            if depth_map is not None and baseline > 0.0:
                col = int(np.round(measurement[0]))
                row = int(np.round(measurement[1]))
                h, w = depth_map.shape
                if 0 <= row < h and 0 <= col < w:
                    stereo_attempt_counts[frame_idx] += 1
                    depth_value = float(depth_map[row, col])
                    if np.isfinite(depth_value) and depth_value > 0.0:
                        disparity = (fx * baseline) / depth_value
                        stereo_measurement = gtsam.StereoPoint2(
                            float(measurement[0]),
                            float(measurement[0] - disparity),
                            float(measurement[1]),
                        )
            if stereo_measurement is not None and frame_idx == birth_frame:
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
                stereo_success_counts[frame_idx] += 1
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
    initial_landmarks = initial_landmarks_world

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
        "raw_observation_counts": raw_observation_counts,
        "gated_observation_counts": gated_observation_counts,
        "stereo_attempt_counts": stereo_attempt_counts,
        "stereo_success_counts": stereo_success_counts,
        "residual_norms": {
            idx: np.asarray(values, dtype=np.float64) for idx, values in residual_norms.items()
        },
        "nan_landmark_indices": np.array(nan_landmark_indices, dtype=int),
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

    frame_order = ba_result["frames_for_ba"]
    residual_norms = ba_result.get("residual_norms", {})
    residual_stats = np.full((len(frame_order), 3), np.nan, dtype=np.float64)
    for idx, frame_idx in enumerate(frame_order):
        values = residual_norms.get(frame_idx)
        if values is None or values.size == 0:
            continue
        residual_stats[idx, 0] = float(np.mean(values))
        residual_stats[idx, 1] = float(np.median(values))
        residual_stats[idx, 2] = float(np.percentile(values, 90.0))

    np.savez(
        RESULTS_PATH,
        frame_ids=frame_ids,
        frame_indices=frame_indices,
        tracking_summary=tracking_summary,
        frames_for_ba=np.array(frame_order, dtype=int),
        landmark_original_indices=np.array(ba_result["landmark_original_indices"], dtype=int),
        reprojection_before=ba_result["reprojection_before"],
        reprojection_after=ba_result["reprojection_after"],
        optimized_translation_errors=optimized_translation_errors,
        optimized_rotation_errors=optimized_rotation_errors,
        optimized_translation_norms=np.linalg.norm(optimized_translation_errors, axis=1),
        optimized_rotation_norms_deg=np.rad2deg(np.linalg.norm(optimized_rotation_errors, axis=1)),
        stereo_counts=np.array([ba_result["stereo_counts"][idx] for idx in ba_result["frames_for_ba"]], dtype=int),
        mono_counts=np.array([ba_result["mono_counts"][idx] for idx in ba_result["frames_for_ba"]], dtype=int),
        raw_observation_counts=np.array(
            [ba_result["raw_observation_counts"][idx] for idx in ba_result["frames_for_ba"]],
            dtype=int,
        ),
        gated_observation_counts=np.array(
            [ba_result["gated_observation_counts"][idx] for idx in ba_result["frames_for_ba"]],
            dtype=int,
        ),
        stereo_attempt_counts=np.array(
            [ba_result["stereo_attempt_counts"][idx] for idx in ba_result["frames_for_ba"]],
            dtype=int,
        ),
        stereo_success_counts=np.array(
            [ba_result["stereo_success_counts"][idx] for idx in ba_result["frames_for_ba"]],
            dtype=int,
        ),
        nan_landmark_indices=ba_result["nan_landmark_indices"],
        residual_stats=residual_stats,
    )
    print(f"Saved bundle adjustment results to {RESULTS_PATH}")


def plot_feature_tracks(
    rectified_frames: list[RectifiedStereoFrame],
    track_history: list[np.ndarray],
) -> None:
    anchor_image = rectified_frames[0].left_rect
    anchor_points = track_history[0]
    valid_anchor = ~np.isnan(anchor_points[:, 0])
    anchor_points_valid = anchor_points[valid_anchor]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].set_title("Anchor frame with FAST keypoints")
    axes[0].imshow(anchor_image)
    if anchor_points_valid.size:
        axes[0].scatter(anchor_points_valid[:, 0], anchor_points_valid[:, 1], s=10, c="lime", linewidths=0.5)
    axes[0].axis("off")

    last_frame_image = rectified_frames[-1].left_rect
    axes[1].set_title("Tracked KLT trajectories")
    axes[1].imshow(last_frame_image)
    cmap = plt.cm.get_cmap("viridis", max(1, anchor_points_valid.shape[0]))
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
) -> None:
    frames = sorted(optimized_pose_dict.keys())

    initial_xyz = np.array([to_numpy_vec3(initial_pose_dict[f].translation()) for f in frames], dtype=np.float64)
    optimized_xyz = np.array([to_numpy_vec3(optimized_pose_dict[f].translation()) for f in frames], dtype=np.float64)
    ground_truth_xyz = np.array(
        [to_numpy_vec3(sequence_sample.relative_pose(0, f).translation()) for f in frames],
        dtype=np.float64,
    )

    plt.figure(figsize=(10, 6))
    plt.title("Trajectory comparison (XY plane)")
    plt.plot(initial_xyz[:, 0], initial_xyz[:, 1], "o--", label="Initial PnP")
    plt.plot(optimized_xyz[:, 0], optimized_xyz[:, 1], "o-", label="Optimized BA")
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
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.show()


def plot_pose_error_norms_per_frame(
    initial_pose_dict: dict[int, gtsam.Pose3],
    optimized_pose_dict: dict[int, gtsam.Pose3],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
) -> None:
    shared_frames = sorted(set(initial_pose_dict.keys()) & set(optimized_pose_dict.keys()))
    if not shared_frames:
        return

    initial_errors = compute_pose_errors_against_ground_truth(initial_pose_dict, sequence_sample)
    optimized_errors = compute_pose_errors_against_ground_truth(optimized_pose_dict, sequence_sample)

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
    axes[0].plot(shared_frames, translation_after, "o-", label="After BA")
    axes[0].set_xlabel("Frame index")
    axes[0].set_ylabel("Translation error (m)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_title("Rotation error norms")
    axes[1].plot(shared_frames, rotation_before_deg, "o--", label="Before BA")
    axes[1].plot(shared_frames, rotation_after_deg, "o-", label="After BA")
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
    )

    rectified_frames = [frame.rectify() for frame in sequence.frames]
    depth_frames = [sgbm.compute_depth(frame, max_depth=MAX_DEPTH) for frame in rectified_frames]

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
            f"Only {filtered_keypoints.shape[0]} keypoints with valid depth after filtering; "
            f"consider relaxing FAST/border/max-depth settings."
        )

    anchor_frame = build_feature_frame(depth_frames[0], attributes)
    track_history, active_indices, _, _ = track_with_klt(gray_sequence, filtered_keypoints.astype(np.float32))
    track_history = [frame.astype(np.float32) for frame in track_history]

    attributes["birth_frames"] = np.zeros(attributes["keypoints"].shape[0], dtype=int)
    track_history, attributes = augment_tracks_with_new_features(
        gray_sequence=gray_sequence,
        depth_frames=depth_frames,
        track_history=track_history,
        attributes=attributes,
    )
    active_indices = compute_active_indices_from_history(track_history)

    sequence_results = estimate_sequence_poses(
        rectified_frames=rectified_frames,
        depth_frames=depth_frames,
        track_history=track_history,
        active_indices=active_indices,
        sequence_sample=sequence,
    )

    tracking_summary = summarise_tracking_results(sequence, track_history, sequence_results)

    ba_result = run_bundle_adjustment(
        rectified_frames=rectified_frames,
        track_history=track_history,
        attributes=attributes,
        sequence_results=sequence_results,
        depth_frames=depth_frames,
    )

    save_results(tracking_summary, ba_result, sequence)

    pose_errors = compute_pose_errors_against_ground_truth(ba_result["optimized_pose_dict"], sequence)
    optimized_translation_norms = [pose_errors[idx]["translation_norm"] for idx in sorted(pose_errors)]
    optimized_rotation_norms_deg = [
        np.rad2deg(pose_errors[idx]["rotation_norm_rad"]) for idx in sorted(pose_errors)
    ]

    rms_before = float(np.sqrt(np.mean(ba_result["reprojection_before"] ** 2)))
    rms_after = float(np.sqrt(np.mean(ba_result["reprojection_after"] ** 2)))

    print("Bundle adjustment summary")
    print(f"  Frames optimised: {ba_result['frames_for_ba']}")
    print(f"  Landmarks optimised: {len(ba_result['landmark_original_indices'])}")
    print(f"  Reprojection RMS: {rms_before:.3f} px -> {rms_after:.3f} px")
    print(
        "  Translation error norms (m):",
        np.array2string(np.asarray(optimized_translation_norms), precision=3),
    )
    print(
        "  Rotation error norms (deg):",
        np.array2string(np.asarray(optimized_rotation_norms_deg), precision=2),
    )
    stereo_counts = ba_result["stereo_counts"]
    mono_counts = ba_result["mono_counts"]
    frame_order = ba_result["frames_for_ba"]
    stereo_per_frame = [stereo_counts[idx] for idx in frame_order]
    mono_per_frame = [mono_counts[idx] for idx in frame_order]
    raw_per_frame = [ba_result["raw_observation_counts"][idx] for idx in frame_order]
    gated_per_frame = [ba_result["gated_observation_counts"][idx] for idx in frame_order]
    stereo_attempts_per_frame = [ba_result["stereo_attempt_counts"][idx] for idx in frame_order]
    stereo_success_per_frame = [ba_result["stereo_success_counts"][idx] for idx in frame_order]
    residual_norms = ba_result["residual_norms"]
    residual_means: list[float] = []
    residual_medians: list[float] = []
    residual_q90: list[float] = []
    for idx in frame_order:
        vals = residual_norms.get(idx, np.empty(0))
        if vals.size:
            residual_means.append(float(np.mean(vals)))
            residual_medians.append(float(np.median(vals)))
            residual_q90.append(float(np.percentile(vals, 90.0)))
        else:
            residual_means.append(float("nan"))
            residual_medians.append(float("nan"))
            residual_q90.append(float("nan"))
    print(
        "  Stereo factors:",
        f"{sum(stereo_per_frame)} total -> {np.array2string(np.asarray(stereo_per_frame), separator=', ')}",
    )
    print(
        "  Mono factors:",
        f"{sum(mono_per_frame)} total -> {np.array2string(np.asarray(mono_per_frame), separator=', ')}",
    )
    print(
        "  Observations (raw→gated):",
        f"{np.array2string(np.asarray(raw_per_frame), separator=', ')} -> "
        f"{np.array2string(np.asarray(gated_per_frame), separator=', ')}",
    )
    print(
        "  Stereo attempts→successes:",
        f"{np.array2string(np.asarray(stereo_attempts_per_frame), separator=', ')} -> "
        f"{np.array2string(np.asarray(stereo_success_per_frame), separator=', ')}",
    )
    print(
        "  Residual stats (mean/median/90th px):",
        f"{np.array2string(np.asarray(residual_means), separator=', ')} / "
        f"{np.array2string(np.asarray(residual_medians), separator=', ')} / "
        f"{np.array2string(np.asarray(residual_q90), separator=', ')}",
    )
    nan_landmarks = ba_result["nan_landmark_indices"]
    if nan_landmarks.size:
        preview = nan_landmarks[:10]
        print(f"  Landmarks with NaN attributes: {nan_landmarks.size} (indices sample {preview.tolist()})")

    plot_feature_tracks(rectified_frames, track_history)
    plot_match_debug(rectified_frames, sequence_results)
    plot_pose_trajectories(
        ba_result["initial_pose_dict"],
        ba_result["optimized_pose_dict"],
        sequence,
        ba_result["initial_landmarks"],
        ba_result["optimized_landmarks"],
        landmark_stride=5,
    )
    plot_pose_error_norms_per_frame(
        ba_result["initial_pose_dict"],
        ba_result["optimized_pose_dict"],
        sequence,
    )
    plot_reprojection_error_histograms(
        ba_result["reprojection_before"],
        ba_result["reprojection_after"],
    )

    all_tracking_results.append(sequence_results)
    all_ba_results.append(ba_result)

# %%
