# %%
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import gtsam
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

# %%
# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
NUM_SEQUENCE_SAMPLES = 1
SEQUENCE_LENGTH = 20
ENVIRONMENT = "AbandonedFactory"
DIFFICULTY = "easy"
TRAJECTORY = "P001"
SAMPLING_MODE = "stride"
MIN_STRIDE = 2
MAX_STRIDE = 2
BASE_SEED = 4

# Feature detection / tracking
MAX_FEATURE_COUNT = 1024
FAST_THRESHOLD = 25
FAST_NONMAX = True
FAST_BORDER = 12

LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)
LK_MIN_EIG_THRESHOLD = 1e-4

MIN_MATCHES_FOR_PNP = 30
MAX_DEPTH = 20.0

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "klt_local_vo_tracking.npz"

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


def pose_error_components(pose: gtsam.Pose3) -> tuple[np.ndarray, np.ndarray]:
    translation = np.asarray(pose.translation(), dtype=np.float64)
    rotation = gtsam.Rot3.Logmap(pose.rotation())
    return translation, rotation


def summarise_results(
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


def save_results(
    summary: dict[str, Any],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
    results: list[dict[str, Any]],
) -> None:
    frame_ids = np.array(sequence_sample.frame_ids, dtype=int)
    frame_indices = np.array(sequence_sample.frame_indices, dtype=int)

    statuses = np.empty(sequence_sample.length, dtype=object)
    statuses[:] = "anchor"
    for result in results:
        statuses[result["frame_index"]] = result["status"]

    np.savez(
        RESULTS_PATH,
        frame_ids=frame_ids,
        frame_indices=frame_indices,
        statuses=statuses,
        **summary,
    )
    print(f"Saved summary data to {RESULTS_PATH}")


def plot_feature_tracks(
    rectified_frames: list[RectifiedStereoFrame],
    track_history: list[np.ndarray],
    results: list[dict[str, Any]],
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

    frame_indices = [res["frame_index"] for res in results if res["status"] == "success"]
    translation_norms = [np.linalg.norm(np.asarray(res["pose_error"].translation())) for res in results if res["status"] == "success"]
    rotation_norms = [
        np.rad2deg(np.linalg.norm(gtsam.Rot3.Logmap(res["pose_error"].rotation())))
        for res in results
        if res["status"] == "success"
    ]

    axes[2].set_title("Pose errors per frame")
    if frame_indices:
        axes[2].plot(frame_indices, translation_norms, marker="o", label="Translation (m)")
        axes[2].plot(frame_indices, rotation_norms, marker="s", label="Rotation (deg)")
    axes[2].set_xlabel("Frame index")
    axes[2].legend()
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


# %%
# ----------------------------------------------------------------------
# Main execution
# ----------------------------------------------------------------------
all_results: list[list[dict[str, Any]]] = []

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

    sequence_results = estimate_sequence_poses(
        anchor_frame=anchor_frame,
        rectified_frames=rectified_frames,
        track_history=track_history,
        active_indices=active_indices,
        sequence_sample=sequence,
    )

    summary = summarise_results(sequence, track_history, sequence_results)
    save_results(summary, sequence, sequence_results)

    plot_feature_tracks(rectified_frames, track_history, sequence_results)
    plot_match_debug(rectified_frames, sequence_results)

    all_results.append(sequence_results)

# %%
