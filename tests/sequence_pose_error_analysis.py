# %%
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import torch
from tqdm import tqdm

import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gtsam  # noqa: E402
from gtsam.symbol_shorthand import X  # noqa: E402
from backend.pose_graph import GtsamPoseGraph  # noqa: E402
from depth.sgbm import SGBM  # noqa: E402
from registration.lighterglue import LighterglueMatcher  # noqa: E402
from registration.registration import FramePair, FramePairWithGroundTruth, StereoFrame  # noqa: E402
from registration.utils import rectify_stereo_frame_pair, solve_pnp  # noqa: E402
import tests.test_utils as test_utils  # noqa: E402

# TODO: validate optimization logic here!

# %%
# Configuration for sampling and output
NUM_SEQUENCE_SAMPLES = 100
SEQUENCE_LENGTH = 10
ENVIRONMENT = "AbandonedFactory"
DIFFICULTY = "easy"
TRAJECTORY = "P001"
MAX_DEGREES = 90.0
MAX_DISTANCE = 10.0
BASE_SEED = 0
MAX_DEPTH = 10.0
SAMPLING_MODE = "stride"
MIN_STRIDE = 1
MAX_STRIDE = 1
LOOP_CLOSING_CHANCE = 0.5
POSE_FACTOR_SIGMAS = np.array(
    [
        np.deg2rad(5.0),
        np.deg2rad(5.0),
        np.deg2rad(5.0),
        0.10,
        0.10,
        0.10,
    ],
    dtype=float,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "sequence_pose_errors.npz"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Instantiate depth estimator and matcher once
sgbm = SGBM(num_disparities=16 * 2, block_size=5, image_color="RGB")
matcher = LighterglueMatcher(
    num_features=4096,
    compile=False,
    device=device,
    use_lighterglue_matching=False,
)


def estimate_relative_pose(pair_with_gt: FramePairWithGroundTruth[StereoFrame]) -> dict[str, Any]:
    rectified_pair = rectify_stereo_frame_pair(pair_with_gt)
    depth_pair = sgbm.compute_depth_pair(rectified_pair, max_depth=MAX_DEPTH)

    first_features, second_features = matcher.detect_features([depth_pair.first, depth_pair.second])
    feature_pair = FramePair(first=first_features, second=second_features)
    matched_pair = matcher.match([feature_pair])[0]

    estimated_pose, matched_pnp_pair = solve_pnp(matched_pair)
    return {
        "estimated_pose": estimated_pose,
        "matched_pair": matched_pnp_pair,
    }


def _make_pair(
    sequence_sample: test_utils.FrameSequenceWithGroundTruth[StereoFrame],
    first_local_idx: int,
    second_local_idx: int,
) -> FramePairWithGroundTruth[StereoFrame]:
    frame_first = sequence_sample.frames[first_local_idx]
    frame_second = sequence_sample.frames[second_local_idx]
    gt_pose = sequence_sample.relative_pose(first_local_idx, second_local_idx)

    pair = FramePairWithGroundTruth[StereoFrame](
        first=frame_first,
        second=frame_second,
        first_T_second=gt_pose,
    )

    first_global_idx = sequence_sample.frame_indices[first_local_idx]
    second_global_idx = sequence_sample.frame_indices[second_local_idx]
    first_frame_id = sequence_sample.frame_ids[first_local_idx]
    second_frame_id = sequence_sample.frame_ids[second_local_idx]

    pair.first_index = first_global_idx  # type: ignore[attr-defined]
    pair.second_index = second_global_idx  # type: ignore[attr-defined]
    pair.first_frame_id = first_frame_id  # type: ignore[attr-defined]
    pair.second_frame_id = second_frame_id  # type: ignore[attr-defined]
    pair.frame_indices = (first_global_idx, second_global_idx)  # type: ignore[attr-defined]
    pair.frame_ids = (first_frame_id, second_frame_id)  # type: ignore[attr-defined]
    return pair


def _ensure_node_initialised(
    pose_graph: GtsamPoseGraph,
    src_idx: int,
    dst_idx: int,
    measured_pose: gtsam.Pose3,
) -> None:
    if not pose_graph.values.exists(X(src_idx)):
        pose_graph.values.insert(X(src_idx), gtsam.Pose3.Identity())
    if pose_graph.values.exists(X(dst_idx)):
        return
    try:
        src_pose = pose_graph.values.atPose3(X(src_idx))
    except RuntimeError:
        src_pose = gtsam.Pose3.Identity()
    pose_graph.values.insert(X(dst_idx), src_pose.compose(measured_pose))


def process_sequence_sample(
    sequence_sample: test_utils.FrameSequenceWithGroundTruth[StereoFrame],
    sample_seed: int,
    sample_idx: int,
) -> list[dict[str, Any]]:
    pose_graph = GtsamPoseGraph(K=sequence_sample.frames[0].calibration.K_left_rect)
    noise_model = gtsam.noiseModel.Diagonal.Sigmas(POSE_FACTOR_SIGMAS)

    measurement_cache: dict[tuple[int, int], dict[str, Any]] = {}
    factor_count = 0

    # First connect consecutive frames to initialise the chain.
    for local_idx in range(sequence_sample.length - 1):
        next_idx = local_idx + 1
        pair = _make_pair(sequence_sample, local_idx, next_idx)
        result = estimate_relative_pose(pair)
        measurement_cache[(local_idx, next_idx)] = result

        _ensure_node_initialised(pose_graph, local_idx, next_idx, result["estimated_pose"])
        pose_graph.add_between_pose_factor(local_idx, next_idx, result["estimated_pose"], noise_model)
        factor_count += 1

    # Add loop-closure style factors between all other frame pairs in the sequence.
    if random.random() < LOOP_CLOSING_CHANCE:
        for first_idx in range(sequence_sample.length):
            for second_idx in range(first_idx + 2, sequence_sample.length):
                try:
                    if (first_idx, second_idx) in measurement_cache:
                        continue
                    pair = _make_pair(sequence_sample, first_idx, second_idx)
                    result = estimate_relative_pose(pair)
                    measurement_cache[(first_idx, second_idx)] = result

                    _ensure_node_initialised(
                        pose_graph, first_idx, second_idx, result["estimated_pose"]
                    )
                    pose_graph.add_between_pose_factor(
                        first_idx, second_idx, result["estimated_pose"], noise_model
                    )
                    factor_count += 1
                except Exception as exc:  # noqa: BLE001
                    failed_samples.append((seed, str(exc)))
                    print(f"Failed to estimate relative pose between {first_idx} and {second_idx}: {exc}")
                    continue

    terminal_local_idx = sequence_sample.length - 1
    if terminal_local_idx > 0 and (0, terminal_local_idx) not in measurement_cache:
        pair = _make_pair(sequence_sample, 0, terminal_local_idx)
        result = estimate_relative_pose(pair)
        measurement_cache[(0, terminal_local_idx)] = result
        _ensure_node_initialised(
            pose_graph, 0, terminal_local_idx, result["estimated_pose"]
        )
        pose_graph.add_between_pose_factor(
            0, terminal_local_idx, result["estimated_pose"], noise_model
        )
        factor_count += 1

    pose_graph.optimize()

    base_pose = sequence_sample.world_poses[0]
    base_frame_idx = sequence_sample.frame_indices[0]
    base_frame_id = sequence_sample.frame_ids[0]

    estimated_pose = pose_graph.values.atPose3(X(terminal_local_idx))
    ground_truth_pose = base_pose.inverse() * sequence_sample.world_poses[terminal_local_idx]
    pose_error = ground_truth_pose.between(estimated_pose)

    translation_error = np.asarray(pose_error.translation())
    rotation_error = gtsam.Rot3.Logmap(pose_error.rotation())
    estimated_translation = estimated_pose.translation()
    estimated_rotation = estimated_pose.rotation()

    current_frame_idx = sequence_sample.frame_indices[terminal_local_idx]
    current_frame_id = sequence_sample.frame_ids[terminal_local_idx]

    inlier_source = measurement_cache.get((0, terminal_local_idx))
    if inlier_source is not None and inlier_source.get("matched_pair") is not None:
        inlier_matches = inlier_source["matched_pair"].matches
        inlier_count = int(len(inlier_matches))
    else:
        inlier_count = 0

    return [
        {
            "seed": sample_seed,
            "sequence_idx": sample_idx,
            "sequence_length": sequence_sample.length,
            "local_index": terminal_local_idx,
            "translation_error": translation_error,
            "rotation_error": rotation_error,
            "translation_norm": float(np.linalg.norm(translation_error)),
            "rotation_norm_rad": float(np.linalg.norm(rotation_error)),
            "rotation_norm_deg": float(np.linalg.norm(np.rad2deg(rotation_error))),
            "inliers": inlier_count,
            "estimated_translation_norm": float(np.linalg.norm(estimated_translation)),
            "estimated_rotation_norm_rad": float(
                np.linalg.norm(gtsam.Rot3.Logmap(estimated_rotation))
            ),
            "ground_truth_translation_norm": float(
                np.linalg.norm(ground_truth_pose.translation())
            ),
            "ground_truth_rotation_norm_rad": float(
                np.linalg.norm(gtsam.Rot3.Logmap(ground_truth_pose.rotation()))
            ),
            "first_index": base_frame_idx,
            "second_index": current_frame_idx,
            "first_frame_id": base_frame_id,
            "second_frame_id": current_frame_id,
            "mid_frame_id": float(base_frame_id + current_frame_id) / 2.0,
            "frame_id_gap": float(current_frame_id - base_frame_id),
            "factor_count": factor_count,
        }
    ]


# %%
# Sample sequences, estimate poses, and collect error metrics
error_records: list[dict[str, Any]] = []
failed_samples: list[tuple[int, str]] = []

for idx in tqdm(range(NUM_SEQUENCE_SAMPLES), desc="Sampling sequence pose estimates"):
    seed = BASE_SEED + idx

    try:
        sequence_sample = test_utils.load_tartanair_sequence_segment(
            env=ENVIRONMENT,
            traj=TRAJECTORY,
            difficulty=DIFFICULTY,
            sequence_length=SEQUENCE_LENGTH,
            seed=seed,
            sampling_mode=SAMPLING_MODE,
            min_stride=MIN_STRIDE,
            max_stride=MAX_STRIDE,
        )
    except Exception as exc:  # noqa: BLE001
        failed_samples.append((seed, f"sequence_sampling_error: {exc}"))
        continue

    try:
        records = process_sequence_sample(sequence_sample, seed, idx)
    except Exception as exc:  # noqa: BLE001
        failed_samples.append((seed, str(exc)))
        continue

    error_records.extend(records)

if not error_records:
    raise RuntimeError("No pose estimates succeeded; cannot compute error statistics.")

first_indices = np.array([record["first_index"] for record in error_records], dtype=int)
second_indices = np.array([record["second_index"] for record in error_records], dtype=int)
first_frame_ids = np.array([record["first_frame_id"] for record in error_records], dtype=int)
second_frame_ids = np.array([record["second_frame_id"] for record in error_records], dtype=int)
mid_frame_ids = np.array([record["mid_frame_id"] for record in error_records], dtype=float)
frame_id_gaps = np.array([record["frame_id_gap"] for record in error_records], dtype=float)
sequence_indices = np.array([record["sequence_idx"] for record in error_records], dtype=int)
local_indices = np.array([record["local_index"] for record in error_records], dtype=int)
sequence_lengths = np.array([record["sequence_length"] for record in error_records], dtype=int)

error_vectors = np.stack(
    [np.concatenate([record["rotation_error"], record["translation_error"]]) for record in error_records],
    axis=0,
)

np.savez(
    RESULTS_PATH,
    error_vectors=error_vectors,
    translation_errors=np.stack([record["translation_error"] for record in error_records], axis=0),
    rotation_errors_rad=np.stack([record["rotation_error"] for record in error_records], axis=0),
    seeds=np.array([record["seed"] for record in error_records], dtype=int),
    sequence_indices=sequence_indices,
    local_indices=local_indices,
    sequence_lengths=sequence_lengths,
    translation_norms=np.array([record["translation_norm"] for record in error_records], dtype=float),
    rotation_norms_rad=np.array([record["rotation_norm_rad"] for record in error_records], dtype=float),
    rotation_norms_deg=np.array([record["rotation_norm_deg"] for record in error_records], dtype=float),
    inlier_counts=np.array([record["inliers"] for record in error_records], dtype=int),
    estimated_translation_norms=np.array(
        [record["estimated_translation_norm"] for record in error_records], dtype=float
    ),
    estimated_rotation_norms_rad=np.array(
        [record["estimated_rotation_norm_rad"] for record in error_records], dtype=float
    ),
    ground_truth_translation_norms=np.array(
        [record["ground_truth_translation_norm"] for record in error_records], dtype=float
    ),
    ground_truth_rotation_norms_rad=np.array(
        [record["ground_truth_rotation_norm_rad"] for record in error_records], dtype=float
    ),
    first_indices=first_indices,
    second_indices=second_indices,
    first_frame_ids=first_frame_ids,
    second_frame_ids=second_frame_ids,
    mid_frame_ids=mid_frame_ids,
    frame_id_gaps=frame_id_gaps,
    factor_counts=np.array([record["factor_count"] for record in error_records], dtype=int),
    failed_samples=np.array(failed_samples, dtype=object),
)

print(f"Saved error data for {len(error_records)} pose estimates to {RESULTS_PATH}")
if failed_samples:
    print(f"{len(failed_samples)} samples failed during processing.")

# %%
# Compute summary statistics and covariance of the error vector
translation_errors = error_vectors[:, 3:]
rotation_errors = error_vectors[:, :3]

translation_norms = np.linalg.norm(translation_errors, axis=1)
rotation_norms_rad = np.linalg.norm(rotation_errors, axis=1)
rotation_norms_deg = np.rad2deg(rotation_norms_rad)

inlier_counts = np.array([record["inliers"] for record in error_records], dtype=float)
factor_counts = np.array([record["factor_count"] for record in error_records], dtype=float)
estimated_translation_norms = np.array(
    [record["estimated_translation_norm"] for record in error_records], dtype=float
)
estimated_rotation_norms_rad = np.array(
    [record["estimated_rotation_norm_rad"] for record in error_records], dtype=float
)
estimated_rotation_norms_deg = np.rad2deg(estimated_rotation_norms_rad)
ground_truth_translation_norms = np.array(
    [record["ground_truth_translation_norm"] for record in error_records], dtype=float
)
ground_truth_rotation_norms_rad = np.array(
    [record["ground_truth_rotation_norm_rad"] for record in error_records], dtype=float
)
ground_truth_rotation_norms_deg = np.rad2deg(ground_truth_rotation_norms_rad)

translation_threshold = 5.0
rotation_threshold_deg = 40.0
estimated_translation_threshold = 5.0
estimated_rotation_threshold_deg = None
inlier_min = 50
# Mask out large motion errors and extremely low inlier support to focus on typical samples
mask = (
    (translation_norms <= translation_threshold)
    & (rotation_norms_deg <= rotation_threshold_deg)
    & (inlier_counts >= inlier_min)
)

if estimated_translation_threshold is not None:
    mask &= estimated_translation_norms <= estimated_translation_threshold

if estimated_rotation_threshold_deg is not None:
    mask &= estimated_rotation_norms_deg <= estimated_rotation_threshold_deg

if not np.any(mask):
    raise RuntimeError("Mask filtered out all samples; relax thresholds to retain data.")

masked_indices = np.nonzero(mask)[0]

# %%
# Temporal visualization of error magnitudes along the trajectory
temporal_mask = mask & np.isfinite(mid_frame_ids)
if np.any(temporal_mask):
    temporal_positions = mid_frame_ids[temporal_mask]
    translation_temporal = translation_norms[temporal_mask]
    rotation_temporal = rotation_norms_deg[temporal_mask]
    frame_gap_temporal = frame_id_gaps[temporal_mask]
    inlier_temporal = inlier_counts[temporal_mask]

    sort_order = np.argsort(temporal_positions)
    temporal_positions_sorted = temporal_positions[sort_order]
    translation_temporal_sorted = translation_temporal[sort_order]
    rotation_temporal_sorted = rotation_temporal[sort_order]
    frame_gap_temporal_sorted = frame_gap_temporal[sort_order]
    inlier_temporal_sorted = inlier_temporal[sort_order]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    scatter_translation = axes[0].scatter(
        temporal_positions_sorted,
        translation_temporal_sorted,
        c=frame_gap_temporal_sorted,
        cmap="viridis",
        alpha=0.8,
    )
    axes[0].plot(temporal_positions_sorted, translation_temporal_sorted, color="tab:blue", alpha=0.4)
    axes[0].set_ylabel("Translation Error Norm (m)")
    axes[0].set_title("Translation Error vs. Frame Index")
    axes[0].grid(alpha=0.2, linestyle="--")

    scatter_rotation = axes[1].scatter(
        temporal_positions_sorted,
        rotation_temporal_sorted,
        c=inlier_temporal_sorted,
        cmap="plasma",
        alpha=0.8,
    )
    axes[1].plot(temporal_positions_sorted, rotation_temporal_sorted, color="tab:orange", alpha=0.4)
    axes[1].set_ylabel("Rotation Error Norm (deg)")
    axes[1].set_xlabel("Mid-point Frame Index")
    axes[1].set_title("Rotation Error vs. Frame Index")
    axes[1].grid(alpha=0.2, linestyle="--")

    plt.colorbar(scatter_translation, ax=axes[0], label="Frame Gap (frames)")
    plt.colorbar(scatter_rotation, ax=axes[1], label="Inlier Count")
    plt.tight_layout()
    plt.show()
else:
    print("Temporal metadata unavailable; skipping temporal visualization.")


def format_vector(vec: np.ndarray) -> str:
    return ", ".join(f"{value:.4f}" for value in vec)


translation_errors_masked = translation_errors[mask]
rotation_errors_masked = rotation_errors[mask]

print("Translation component mean (m):", format_vector(translation_errors_masked.mean(axis=0)))
print("Translation component std (m):", format_vector(translation_errors_masked.std(axis=0)))
print("Rotation component mean (rad):", format_vector(rotation_errors_masked.mean(axis=0)))
print("Rotation component std (rad):", format_vector(rotation_errors_masked.std(axis=0)))

translation_norms_masked = translation_norms[mask]
rotation_norms_deg_masked = rotation_norms_deg[mask]

print(f"Translation norm mean: {translation_norms_masked.mean():.4f} m")
print(f"Translation norm median: {np.median(translation_norms_masked):.4f} m")
print(f"Translation norm std: {translation_norms_masked.std():.4f} m")
print(f"Rotation norm mean: {rotation_norms_deg_masked.mean():.4f} deg")
print(f"Rotation norm median: {np.median(rotation_norms_deg_masked):.4f} deg")
print(f"Rotation norm std: {rotation_norms_deg_masked.std():.4f} deg")

covariance = np.cov(error_vectors[mask], rowvar=False)
print("Error vector covariance matrix shape:", covariance.shape)

np.set_printoptions(suppress=True, precision=6)
print(covariance)

# %%
# Visualize error distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(translation_norms[mask], bins=10, color="tab:blue", alpha=0.7)
axes[0].set_title("Translation Error Norm (m)")
axes[0].set_xlabel("Meters")
axes[0].set_ylabel("Frequency")

axes[1].hist(rotation_norms_deg[mask], bins=10, color="tab:orange", alpha=0.7)
axes[1].set_title("Rotation Error Norm (deg)")
axes[1].set_xlabel("Degrees")

plt.tight_layout()
plt.show()

# %%
# Augmented covariance analysis including inliers and pose magnitude
inlier_counts_col = inlier_counts[mask].reshape(-1, 1)
factor_counts_col = factor_counts[mask].reshape(-1, 1)
estimated_translation_norms_col = estimated_translation_norms[mask].reshape(-1, 1)
estimated_rotation_norms_rad_col = estimated_rotation_norms_rad[mask].reshape(-1, 1)

augmented_metrics = np.hstack(
    [
        error_vectors[mask],
        inlier_counts_col,
        factor_counts_col,
        estimated_translation_norms_col,
        estimated_rotation_norms_rad_col,
    ]
)

augmented_labels = [
    "err_tx",
    "err_ty",
    "err_tz",
    "err_rx",
    "err_ry",
    "err_rz",
    "inliers",
    "factor_count",
    "est_trans_norm",
    "est_rot_norm_rad",
]

augmented_cov = np.cov(augmented_metrics, rowvar=False)
augmented_corr = np.corrcoef(augmented_metrics, rowvar=False)

print("Augmented covariance matrix (includes inliers & pose magnitude):")
for row_label, row in zip(augmented_labels, augmented_cov):
    formatted = "  ".join(f"{value: .4e}" for value in row)
    print(f"{row_label:>16}: {formatted}")

print("\nAugmented correlation matrix:")
for row_label, row in zip(augmented_labels, augmented_corr):
    formatted = "  ".join(f"{value: .4f}" for value in row)
    print(f"{row_label:>16}: {formatted}")

# %%
# Visualization of correlations between support metrics and error magnitudes
translation_norms_clipped = translation_norms[mask]
inliers_clipped = inlier_counts[mask]
est_trans_norms_clipped = estimated_translation_norms[mask]
rotation_norms_deg_clipped = rotation_norms_deg[mask]
est_rot_norms_deg_clipped = estimated_rotation_norms_deg[mask]
factor_counts_clipped = factor_counts[mask]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(inliers_clipped, translation_norms_clipped, alpha=0.7, color="tab:blue")
axes[0].set_xlabel("Inlier Count")
axes[0].set_ylabel("Translation Error Norm (m)")
axes[0].set_title("Error vs Inlier Count")

scatter_est_trans = axes[1].scatter(
    est_trans_norms_clipped,
    translation_norms_clipped,
    c=factor_counts_clipped,
    alpha=0.7,
    cmap="viridis",
)
axes[1].set_xlabel("Estimated Translation Norm (m)")
axes[1].set_ylabel("Translation Error Norm (m)")
axes[1].set_title("Error vs Estimated Distance")

scatter_est_rot = axes[2].scatter(
    est_rot_norms_deg_clipped,
    rotation_norms_deg_clipped,
    c=factor_counts_clipped,
    alpha=0.7,
    cmap="viridis",
)
axes[2].set_xlabel("Estimated Rotation Norm (deg)")
axes[2].set_ylabel("Rotation Error Norm (deg)")
axes[2].set_title("Error vs Estimated Rotation")

plt.colorbar(scatter_est_trans, ax=axes[1], label="Factor Count")
plt.colorbar(scatter_est_rot, ax=axes[2], label="Factor Count")

plt.tight_layout()
plt.show()

# %%
# Explained variance for translation and rotation error norms
def compute_explained_variance(target: np.ndarray, *feature_columns: np.ndarray) -> float:
    if len(feature_columns) == 0:
        raise ValueError("At least one feature column is required.")
    X_design = np.column_stack([np.ones(len(target), dtype=float)] + list(feature_columns))
    coeffs, *_ = np.linalg.lstsq(X_design, target, rcond=None)
    y_pred = X_design @ coeffs
    residual = target - y_pred
    total_variance = target - target.mean()
    sse = float(residual @ residual)
    sst = float(total_variance @ total_variance)
    return float("nan") if sst == 0 else 1.0 - sse / sst


feature_sources = {
    "inliers": inlier_counts[mask],
    "factor_count": factor_counts[mask],
    "est_trans_norm": estimated_translation_norms[mask],
    "est_rot_norm_deg": estimated_rotation_norms_deg[mask],
    "gt_trans_norm": ground_truth_translation_norms[mask],
    "gt_rot_norm_deg": ground_truth_rotation_norms_deg[mask],
}

# %%
# Clustered covariance by inlier count and motion magnitude
def assign_bins(values: np.ndarray, quantiles: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    if np.allclose(values.min(), values.max()):
        return np.zeros_like(values, dtype=int), np.array([values.min(), values.max()], dtype=float)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if edges.size == 0:
        return np.zeros_like(values, dtype=int), np.array([values.min(), values.max()], dtype=float)
    bins = np.digitize(values, edges)
    bounds = np.concatenate(([values.min()], edges, [values.max()])).astype(float)
    bounds[0] = values.min()
    bounds[-1] = values.max()
    return bins.astype(int), bounds


def describe_bin(bin_idx: int, bounds: np.ndarray) -> str:
    lower = bounds[bin_idx]
    upper = bounds[bin_idx + 1]
    if np.isclose(lower, upper):
        return f"{lower:.3f}"
    return f"[{lower:.3f}, {upper:.3f}]"


error_vectors_masked = error_vectors[mask]
translation_errors_masked = translation_errors[mask]
rotation_errors_masked = rotation_errors[mask]

if error_vectors_masked.shape[0] < 6:
    print("Not enough samples after masking to build covariance bins.")
else:
    inlier_bins, inlier_bounds = assign_bins(inliers_clipped, (0.5,))
    trans_bins, trans_bounds = assign_bins(est_trans_norms_clipped, (0.5,))
    rot_bins, rot_bounds = assign_bins(est_rot_norms_deg_clipped, (0.5,))

    cluster_records = []
    bin_keys = np.column_stack([inlier_bins, trans_bins, rot_bins])

    for bin_tuple in np.unique(bin_keys, axis=0):
        mask_cluster = (
            (inlier_bins == bin_tuple[0])
            & (trans_bins == bin_tuple[1])
            & (rot_bins == bin_tuple[2])
        )
        idx = np.nonzero(mask_cluster)[0]
        if idx.size < 3:
            continue
        cluster_errors = error_vectors_masked[idx]
        cov6 = np.cov(cluster_errors, rowvar=False)
        mean6 = cluster_errors.mean(axis=0)
        cluster_records.append(
            {
                "label": (
                    f"Inliers {describe_bin(bin_tuple[0], inlier_bounds)}, "
                    f"Dist {describe_bin(bin_tuple[1], trans_bounds)}, "
                    f"Rot {describe_bin(bin_tuple[2], rot_bounds)}"
                ),
                "cov": cov6,
                "mean": mean6,
                "count": idx.size,
                "indices": idx,
            }
        )

    cluster_records.sort(key=lambda r: r["count"], reverse=True)
    max_clusters = 5
    cluster_records = cluster_records[:max_clusters]

    print(f"Computed covariance for {len(cluster_records)} clusters (top {max_clusters}).")
    for record in cluster_records:
        print(f"{record['label']} -> n={record['count']}, diag={np.diag(record['cov'])[:3]}")

    color_map = plt.get_cmap("tab10")(np.linspace(0, 1, len(cluster_records))) if cluster_records else []

    def plot_covariance_ellipses(ax, clusters, colors, dim_slice, title):
        handles = []
        labels = []
        for color, record in zip(colors, clusters):
            samples = error_vectors_masked[record["indices"]][:, dim_slice]
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                color=color,
                alpha=0.4,
                s=15,
            )
            cov = record["cov"][dim_slice, :][:, dim_slice]
            if np.any(np.isnan(cov)) or cov.shape != (2, 2):
                continue
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                continue
            eigvals = np.maximum(eigvals, 0.0)
            width, height = 2.0 * np.sqrt(eigvals)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            center = record["mean"][dim_slice]
            ellipse = Ellipse(
                xy=center,
                width=width,
                height=height,
                angle=angle,
                edgecolor=color,
                facecolor="none",
                lw=2,
            )
            ax.add_patch(ellipse)
            ax.plot(center[0], center[1], marker="o", color=color, markersize=4)
            handles.append(ellipse)
            labels.append(f"{record['label']} (n={record['count']})")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axhline(0.0, color="gray", linewidth=0.5)
        ax.axvline(0.0, color="gray", linewidth=0.5)
        ax.set_aspect("equal")
        return handles, labels

    if cluster_records:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        handles_xy, labels_xy = plot_covariance_ellipses(
            axes[0],
            cluster_records,
            color_map,
            slice(0, 2),
            "Translation Covariance Ellipses (X/Y)",
        )
        _, _ = plot_covariance_ellipses(
            axes[1],
            cluster_records,
            color_map,
            slice(3, 5),
            "Rotation Covariance Ellipses (R_x/R_y)",
        )
        if handles_xy:
            fig.legend(
                handles_xy,
                labels_xy,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.04),
                ncol=min(len(labels_xy), 3),
                fontsize=9,
            )
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()

target_metrics = {
    "translation_error_norm_m": translation_norms[mask],
    "rotation_error_norm_deg": rotation_norms_deg[mask],
}

combinations = [
    ("inliers",),
    ("factor_count",),
    ("est_trans_norm",),
    ("est_rot_norm_deg",),
    ("gt_trans_norm",),
    ("gt_rot_norm_deg",),
    ("inliers", "est_trans_norm"),
    ("inliers", "est_rot_norm_deg"),
    ("factor_count", "est_trans_norm"),
    ("factor_count", "est_rot_norm_deg"),
    ("est_trans_norm", "est_rot_norm_deg"),
    ("inliers", "factor_count", "est_trans_norm"),
    ("inliers", "factor_count", "est_rot_norm_deg"),
    ("inliers", "factor_count", "est_trans_norm", "est_rot_norm_deg"),
    (
        "inliers",
        "factor_count",
        "est_trans_norm",
        "est_rot_norm_deg",
        "gt_trans_norm",
        "gt_rot_norm_deg",
    ),
]

print("Explained variance (R^2) for error norms:")
for target_name, target_values in target_metrics.items():
    print(f"\nTarget: {target_name}")
    for combo in combinations:
        features = [feature_sources[name] for name in combo]
        r2 = compute_explained_variance(target_values, *features)
        feature_label = ", ".join(combo)
        print(f"  Features [{feature_label:>40}]: R^2 = {r2: .4f}")

# %%
