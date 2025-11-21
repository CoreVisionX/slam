# %%
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
import gtsam
import matplotlib.pyplot as plt
import numpy as np
import torch
import tartanair as ta
import rerun as rr
from tqdm import tqdm
from hydra import compose, initialize
from hydra.utils import instantiate        
from hydra import initialize_config_dir
from gtsam.symbol_shorthand import L
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slam.local_vo.bundle_adjustment import finite_difference_velocity
from viz import rr_log_pose, rr_log_trajectory

from depth.sgbm import SGBM  # noqa: E402
from registration.registration import (  # noqa: E402
    FramePair,
    RectifiedStereoFrame,
    StereoCalibration,
    StereoDepthFrame,
    StereoFrame
)
from registration.utils import draw_matches  # noqa: E402
import tests.test_utils as test_utils  # noqa: E402
from tests.test_utils import se3_inverse
from slam.local_vo import (  # noqa: E402
    BundleAdjustmentConfig,
    FeatureTrack,
    FixedLagBundleAdjuster,
    IncrementalBundleAdjuster,
    KLTFeatureTracker,
    RelativePnPInitializer,
    TrackObservation,
)
from slam.vio.imu_preintegration import ImuPreintegrationConfig
from slam.vio.core import VIO
from tests.vio_example import VIOConfig
from tests.datasets.pipeline import SequencePreprocessor

# TODO: expressing disparity uncertainty for depth measurements in BA properly might help a ton

# %%
# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
NUM_SEQUENCE_SAMPLES = 1
BASE_SEED = 13

USE_IMU_FACTORS = True
DEPTH_MODE = os.environ.get("LOCAL_VO_DEPTH_MODE", "sgbm").strip().lower() # TODO: figure out why on hospital true depth is so much better than sgbm, where does it get messed up?
# maybe try ML based depth estimation, might be the easiest way to get better perf tbh when true depth is getting 5x better results


IMU_GRAVITY_MAGNITUDE = 9.80665
# IMU_ACCEL_NOISE = 0.8e-3
# IMU_GYRO_NOISE = np.deg2rad(0.0006)
# IMU_INTEGRATION_NOISE = 5e-4
# IMU_VELOCITY_PRIOR_SIGMA = 0.1
IMU_BIAS_PRIOR_SIGMAS = np.array(
    [
        3.0000e-3,
        3.0000e-3,
        3.0000e-3,
        1.9393e-05,
        1.9393e-05,
        1.9393e-05,
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
IMU_VELOCITY_PRIOR_SIGMA = 10.0
# IMU_BIAS_PRIOR_SIGMAS = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01], dtype=float) * 100.0

# IMU_GRAVITY_VECTOR = np.array([0.0, 0.0, IMU_GRAVITY_MAGNITUDE], dtype=float)
IMU_GRAVITY_VECTOR = np.array([0.0, 0.0, -IMU_GRAVITY_MAGNITUDE], dtype=float)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "klt_local_vo_bundle_adjustment.npz"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_local_vo_config() -> Any:
    config_dir = Path(__file__).parent / "config"
    if initialize_config_dir is not None:
        with initialize_config_dir(
            config_dir=str(config_dir),
            job_name="local_vo_bundle_adjustment",
            version_base=None,
        ):
            return compose(config_name="local_vo")
    with initialize(
        config_path=str(config_dir),
        job_name="local_vo_bundle_adjustment",
        version_base=None,
    ):
        return compose(config_name="local_vo")


LOCAL_VO_CFG = _load_local_vo_config()
KLT_TRACKER: KLTFeatureTracker = instantiate(LOCAL_VO_CFG.klt_tracker)
RELATIVE_POSE_INITIALIZER: RelativePnPInitializer = instantiate(LOCAL_VO_CFG.relative_pose_initializer)
bundle_adjuster_cfg = OmegaConf.to_container(LOCAL_VO_CFG.bundle_adjuster, resolve=True)
if isinstance(bundle_adjuster_cfg, dict):
    bundle_adjuster_cfg.pop("_target_", None)
    bundle_adjuster_config = BundleAdjustmentConfig(**bundle_adjuster_cfg)
else:
    bundle_adjuster_config = BundleAdjustmentConfig()
BUNDLE_ADJUSTER: FixedLagBundleAdjuster = FixedLagBundleAdjuster(bundle_adjuster_config)
DATA_PIPELINE: SequencePreprocessor = instantiate(LOCAL_VO_CFG.data_pipeline, _convert_="partial")

TRACKING_MAX_DEPTH = float(getattr(getattr(KLT_TRACKER, "config", None), "max_depth", 40.0))
PNP_MAX_DEPTH = float(getattr(getattr(RELATIVE_POSE_INITIALIZER, "config", None), "max_depth", 40.0))
MAX_DEPTH = float(max(TRACKING_MAX_DEPTH, PNP_MAX_DEPTH))


def _frame_timestamp(frame_idx: int, timestamps: np.ndarray | None) -> float:
    if timestamps is None:
        return float(frame_idx)
    if 0 <= frame_idx < len(timestamps):
        return float(timestamps[frame_idx])
    return float(frame_idx)


def _pose_dict_to_tum_array(
    pose_dict: dict[int, gtsam.Pose3],
    frame_timestamps: np.ndarray | None,
) -> np.ndarray:
    if not pose_dict:
        return np.empty((0, 8), dtype=np.float64)

    rows: list[list[float]] = []
    for frame_idx in sorted(pose_dict):
        timestamp = _frame_timestamp(frame_idx, frame_timestamps)
        pose = pose_dict[frame_idx]
        translation = to_numpy_vec3(pose.translation())
        quat = pose.rotation().toQuaternion()
        rows.append(
            [
                timestamp,
                float(translation[0]),
                float(translation[1]),
                float(translation[2]),
                float(quat.x()),
                float(quat.y()),
                float(quat.z()),
                float(quat.w()),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def _write_tum_file(path: Path, tum_data: np.ndarray) -> None:
    if tum_data.size == 0:
        return
    np.savetxt(path, tum_data, fmt="%.9f")
    print(f"Saved TUM trajectory: {path}")


def save_variant_tum_files(
    *,
    sequence_label: str,
    sample_index: int,
    depth_label: str,
    estimated_pose_dict: dict[int, gtsam.Pose3],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
) -> None:
    if not estimated_pose_dict:
        return

    prefix = str(sequence_label)
    est_path = RESULTS_DIR / f"{prefix}_estimated.txt"
    est_tum = _pose_dict_to_tum_array(estimated_pose_dict, sequence_sample.frame_timestamps)
    _write_tum_file(est_path, est_tum)

    gt_pose_dict = {
        idx: sequence_sample.world_poses[idx]
        for idx in estimated_pose_dict
        if 0 <= idx < len(sequence_sample.world_poses)
    }
    gt_path = RESULTS_DIR / f"{prefix}_ground_truth.txt"
    gt_tum = _pose_dict_to_tum_array(gt_pose_dict, sequence_sample.frame_timestamps)
    _write_tum_file(gt_path, gt_tum)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Primary depth variant: {DEPTH_MODE}")

def create_preintegration_params(sequence=None, imu_config: ImuPreintegrationConfig | None = None) -> gtsam.PreintegrationParams:
    if imu_config is None:
        # Fallback to globals if no config provided (for backward compatibility or other tests)
        gravity = IMU_GRAVITY_MAGNITUDE
        accel_noise = IMU_ACCEL_NOISE
        gyro_noise = IMU_GYRO_NOISE
        integration_noise = IMU_INTEGRATION_NOISE
    else:
        gravity = imu_config.gravity_magnitude
        accel_noise = imu_config.accel_noise
        gyro_noise = imu_config.gyro_noise
        integration_noise = imu_config.integration_noise

    params = gtsam.PreintegrationParams.MakeSharedU(gravity)
    # params.n_gravity = IMU_GRAVITY_VECTOR if sequence is None else test_utils.estimate_world_gravity_from_first_batch(sequence)
    if imu_config is not None:
         params.n_gravity = np.array([0.0, 0.0, -gravity]) # Assuming gravity is aligned with Z down
    else:
         params.n_gravity = IMU_GRAVITY_VECTOR
         
    params.setAccelerometerCovariance(np.eye(3) * (accel_noise ** 2))
    params.setGyroscopeCovariance(np.eye(3) * (gyro_noise ** 2))
    # params.setAccelerometerBiasCovariance(np.eye(3) * (LOCAL_VO_CFG.bundle_adjuster.imu_bias_prior_sigmas[0] ** 2))
    # params.setBiasOmegaCovariance(np.eye(3) * (LOCAL_VO_CFG.bundle_adjuster.imu_bias_prior_sigmas[3] ** 2))
    params.setIntegrationCovariance(np.eye(3) * (integration_noise ** 2))
    return params


def _integrate_imu_batch(
    pim: gtsam.PreintegratedImuMeasurements,
    batch: test_utils.FrameImuMeasurements,
) -> None:
    if batch is None:
        raise ValueError("IMU batch missing while attempting to integrate measurements.")

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


def preintegrate_between_frames(sequence, start_idx, end_idx, params, bias, time_offset: float = 0.0, use_combined: bool = False):
    if end_idx <= start_idx:
        raise ValueError("end_idx must be greater than start_idx")

    pim_cls = gtsam.PreintegratedCombinedMeasurements if use_combined else gtsam.PreintegratedImuMeasurements
    pim = pim_cls(params, bias)

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
        dt_diff = abs(pim.deltaTij() - dt_frames)
        if dt_diff > 5e-3:
            print(f"[warn] IMU Δt {pim.deltaTij():.6f} vs expected {dt_frames:.6f}")

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


def _build_initial_pose_dict_from_pnp(
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
    sequence_results: Sequence[dict[str, Any]],
) -> dict[int, gtsam.Pose3]:
    """Create an initial pose dictionary using the incremental PnP estimates."""
    first_pose = gtsam.Pose3(sequence_sample.world_poses[0].rotation(), np.zeros(3))
    pose_dict: dict[int, gtsam.Pose3] = {0: clone_pose(first_pose)}
    pose_lookup: dict[int, gtsam.Pose3] = {}
    for result in sequence_results:
        if result.get("status") != "success":
            continue
        estimated_pose = result.get("estimated_pose")
        if estimated_pose is not None:
            pose_lookup[result["frame_index"]] = clone_pose(estimated_pose)

    last_pose = pose_dict[0]
    for idx in range(1, sequence_sample.length):
        if idx in pose_lookup:
            last_pose = pose_lookup[idx]
        pose_dict[idx] = clone_pose(last_pose)

    return pose_dict


def _initialize_fixed_lag_landmarks(
    bundle_adjuster: FixedLagBundleAdjuster,
    *,
    tracks: Mapping[int, FeatureTrack],
    rectified_frames: Sequence[RectifiedStereoFrame],
    pose_dict: Mapping[int, gtsam.Pose3],
) -> None:
    """Seed landmark positions so smart factors can gate observations."""
    if bundle_adjuster.smoothed_values is None or bundle_adjuster.full_values is None:
        return

    for track_id, track in tracks.items():
        anchor_pose = pose_dict.get(track.anchor_frame)
        if anchor_pose is None:
            continue
        imu_from_left = rectified_frames[track.anchor_frame].calibration.imu_from_left
        anchor_point = gtsam.Point3(*track.anchor_point3.tolist())
        world_point = anchor_pose.compose(imu_from_left).transformFrom(anchor_point)
        key = L(track_id)
        if not bundle_adjuster.full_values.exists(key):
            bundle_adjuster.full_values.insert(key, world_point)
        if not bundle_adjuster.smoothed_values.exists(key):
            bundle_adjuster.smoothed_values.insert(key, world_point)


def _count_observation_types(
    track_history: Sequence[Mapping[int, TrackObservation]],
) -> tuple[dict[int, int], dict[int, int]]:
    stereo_counts: dict[int, int] = {}
    mono_counts: dict[int, int] = {}
    for frame_idx, observations in enumerate(track_history):
        stereo = 0
        mono = 0
        for obs in observations.values():
            depth = float(obs.depth)
            if np.isfinite(depth) and depth > 0.0:
                stereo += 1
            else:
                mono += 1
        stereo_counts[frame_idx] = stereo
        mono_counts[frame_idx] = mono
    return stereo_counts, mono_counts


def run_fixed_lag_adjustment(
    *,
    bundle_adjuster: FixedLagBundleAdjuster,
    rectified_frames: Sequence[RectifiedStereoFrame],
    depth_frames: Sequence[StereoDepthFrame],
    track_history: Sequence[Mapping[int, TrackObservation]],
    tracks: Mapping[int, FeatureTrack],
    sequence_results: Sequence[dict[str, Any]],
    sequence_sample: test_utils.FrameSequenceWithGroundTruth,  # type: ignore[type-arg]
    use_imu: bool = True,
) -> dict[str, Any]:
    if not rectified_frames:
        raise ValueError("No frames provided for fixed-lag optimization.")

    # Construct VIO Config from the first frame's calibration
    calib = rectified_frames[0].calibration
    # Calculate baseline from imu_from_left and imu_from_right
    # Assuming imu_from_right = imu_from_left * T_S0_from_S1
    # T_S0_from_S1 = inv(imu_from_left) * imu_from_right
    # baseline is usually -Tx of T_S1_from_S0 (or Tx of T_S0_from_S1 if right is S1)
    # Let's use the norm of the translation between cameras
    
    # imu_from_left is gtsam.Pose3 in EuRoCStereoCalibration
    imu_from_left_mat = calib.imu_from_left.matrix()
    imu_from_right_mat = calib.imu_from_right.matrix()
    
    T_left_inv = se3_inverse(imu_from_left_mat)
    T_right_from_left = T_left_inv @ imu_from_right_mat
    baseline = np.linalg.norm(T_right_from_left[:3, 3])

    # vio_config = VIOConfig(
    #     gravity=IMU_GRAVITY_VECTOR,
    #     imu_from_left=imu_from_left_mat,
    #     imu_from_right=imu_from_right_mat,
    #     baseline=baseline,
    #     K_left_rect=calib.K_left_rect,
    #     K_right_rect=calib.K_right_rect,
    #     width=calib.width,
    #     height=calib.height,
    #     optimize_every=4, # Hardcoded to match previous logic or config
    # )

    # # Instantiate VIO
    # # We use the global KLT_TRACKER and RELATIVE_POSE_INITIALIZER
    # # Note: VIO will reset them, so the previous tracking results in run_depth_variant might be invalidated
    # # if we were relying on the tracker state, but we passed track_history and tracks as args.
    # vio = VIO(
    #     config=vio_config,
    #     feature_tracker=KLT_TRACKER,
    #     relative_pose_initializer=RELATIVE_POSE_INITIALIZER,
    #     ba=bundle_adjuster
    # )

    # Instansiate VIO using it's hydra config
    config_dir = Path(__file__).parent / "config"
    with initialize_config_dir(
            config_dir=str(config_dir),
            job_name="local_vo_bundle_adjustment",
            version_base=None,
        ):
        cfg = compose("vio_config")
        vio = instantiate(cfg.vio)
        imu_preintegrator = instantiate(cfg.imu_preintegrator)

    vio_config = vio.config

    bundle_adjuster = vio.ba

    frames_for_ba = list(range(len(rectified_frames)))
    
    # Initial state
    first_ts = _frame_timestamp(0, sequence_sample.frame_timestamps)
    first_pose = sequence_sample.world_poses[0]
    first_velocity = finite_difference_velocity(
        sequence_sample.world_poses[0], 
        sequence_sample.world_poses[1], 
        _frame_timestamp(1, sequence_sample.frame_timestamps) - first_ts
    )

    # Reset VIO
    vio.reset(
        timestamp=first_ts,
        left_rect=rectified_frames[0].left_rect,
        right_rect=rectified_frames[0].right_rect,
        t=to_numpy_vec3(first_pose.translation()),
        R=first_pose.rotation().matrix(),
        v=first_velocity,
    )

    # Re-build initial pose dict to match return signature (though VIO doesn't expose it directly, 
    # we can use the one passed in or reconstruct it if needed. 
    # For now, let's use the one from the arguments which was computed by run_depth_variant)
    pose_initials = _build_initial_pose_dict_from_pnp(sequence_sample, sequence_results)
    initial_pose_dict = {idx: clone_pose(pose_initials[idx]) for idx in frames_for_ba}

    params = create_preintegration_params(sequence_sample, imu_preintegrator.config)
    bias = gtsam.imuBias.ConstantBias()

    stereo_counts, mono_counts = _count_observation_types(track_history)
    
    for frame_idx in tqdm(range(1, len(frames_for_ba)), desc="Running bundle adjustment"):
        ts = _frame_timestamp(frame_idx, sequence_sample.frame_timestamps)
        
        # IMU integration
        # VIO expects pim from previous frame to current frame.
        # For frame 0, pim should probably be empty or None?
        # VIO.process takes pim.
        # In the original loop:
        # pim = gtsam.PreintegratedImuMeasurements(params, bias)
        # _integrate_imu_batch(pim, sequence_sample.imu_measurements[frame_idx])
        # This integrates measurements *associated* with frame_idx (usually between prev and curr).
        
        pim = gtsam.PreintegratedImuMeasurements(params, bias)
        if frame_idx < len(sequence_sample.imu_measurements):
             _integrate_imu_batch(pim, sequence_sample.imu_measurements[frame_idx])

        print(f"K_left_rect: {rectified_frames[frame_idx].calibration.K_left_rect}")
        print(f"K_right_rect: {rectified_frames[frame_idx].calibration.K_right_rect}")
        print(f"Baseline: {rectified_frames[frame_idx].calibration.T}")
        print(f"imu_from_left: {rectified_frames[frame_idx].calibration.imu_from_left}")
        print(f"imu_from_right: {rectified_frames[frame_idx].calibration.imu_from_right}")
        
        vio.process(
            timestamp=ts,
            left_rect=rectified_frames[frame_idx].left_rect,
            right_rect=rectified_frames[frame_idx].right_rect,
            pim=pim,
        )

    optimized_pose_list = bundle_adjuster.get_trajectory()
    optimized_pose_dict = {
        idx: optimized_pose_list[idx] for idx in range(min(len(optimized_pose_list), len(frames_for_ba)))
    }

    landmark_indices = sorted(tracks.keys())
    return {
        "frames_for_ba": frames_for_ba,
        "landmark_original_indices": landmark_indices,
        "initial_pose_dict": initial_pose_dict,
        "optimized_pose_dict": optimized_pose_dict,
        "initial_landmarks": [],
        "optimized_landmarks": [],
        "observation_matrix": [],
        "stereo_counts": stereo_counts,
        "mono_counts": mono_counts,
        "imu_bias_before": None,
        "imu_bias_after": None,
    }


def integrate_inertial_trajectory(
    sequence: test_utils.FrameSequenceWithGroundTruth[StereoFrame],  # type: ignore[type-arg]
) -> list[gtsam.Pose3] | None:
    gt_world_to_first = sequence.world_poses[0]

    params = create_preintegration_params(sequence)
    bias = gtsam.imuBias.ConstantBias()

    nav_states = [gtsam.NavState(gtsam.Pose3(gt_world_to_first.rotation(), np.zeros(3)), sequence.imu_measurements[0].world_velocity)]
    translation_gt_world_to_first = gtsam.Pose3(gtsam.Rot3.Identity(), gt_world_to_first.translation()).inverse()
   
    # # preintegrate relative to the previous preintegrated pose
    for idx in range(1, sequence.length):
        batch = sequence.imu_measurements[idx]

        nav_state = nav_states[-1]

        # only reset every few frames
        if idx % 9 == 0 and idx > 1:
            prev_prev_pose = sequence.world_poses[idx - 2]
            prev_pose = sequence.world_poses[idx - 1]
            prev_velocity = (prev_pose.translation() - prev_prev_pose.translation()) / (sequence.frame_timestamps[idx - 1] - sequence.frame_timestamps[idx - 2])
            nav_state = gtsam.NavState(translation_gt_world_to_first * sequence.world_poses[idx - 1], prev_velocity)

        pim = gtsam.PreintegratedImuMeasurements(params, bias)
        _integrate_imu_batch(pim, batch)
        nav_states.append(pim.predict(nav_state, bias))
        
    inertial_poses = [nav_state.pose() for nav_state in nav_states]

    return inertial_poses


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


def _log_bundle_adjuster_bias_diagnostics(result: dict[str, Any], label: str | None) -> None:
    if not result:
        return
    bias_before = result.get("imu_bias_before")
    bias_after = result.get("imu_bias_after")
    if bias_before is None or bias_after is None:
        return
    log_label = label or "bundle_adjustment"
    log_imu_bias_values(bias_before, bias_after, label=log_label)
    plot_imu_bias_values(bias_before, bias_after, label=log_label)


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

    plt.figure(figsize=(15, 9))
    plt.title("Trajectory comparison (XZ plane)")
    # plt.plot(initial_xyz[:, 0], initial_xyz[:, 2], "o--", label=initial_label)
    # plt.plot(optimized_xyz[:, 0], optimized_xyz[:, 2], "o-", label=optimized_label)
    # if optimized_xyz_no_imu is not None:
        # plt.plot(optimized_xyz_no_imu[:, 0], optimized_xyz_no_imu[:, 2], "o-", label="BA w/o IMU")
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
            # plt.plot(additional_xyz[:, 0], additional_xyz[:, 2], "o--", label=label)
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
            # plt.plot(additional_xyz[:, 0], additional_xyz[:, 2], "o-", label=label)
    if inertial_xyz is not None:
        plt.plot(
            inertial_xyz[:, 0],
            inertial_xyz[:, 2],
            "s--",
            label="IMU-only (per-frame)",
        )
    plt.plot(ground_truth_xyz[:, 0], ground_truth_xyz[:, 2], "x-", label="Ground truth")

    if initial_landmarks:
        init_landmarks_arr = np.asarray(initial_landmarks, dtype=np.float64)
        if landmark_stride > 1:
            init_landmarks_arr = init_landmarks_arr[::landmark_stride]
        plt.scatter(
            init_landmarks_arr[:, 0],
            init_landmarks_arr[:, 2],
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
            opt_landmarks_arr[:, 2],
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
    if len(rectified_frames) != len(depth_frames):
        raise ValueError("Number of rectified frames and depth frames must match.")

    KLT_TRACKER.reset()
    RELATIVE_POSE_INITIALIZER.reset_with_gt(sequence_sample)
    track_history: list[dict[int, TrackObservation]] = []
    sequence_results: list[dict[str, Any]] = []

    for frame_idx, (rect_frame, depth_frame) in enumerate(zip(rectified_frames, depth_frames)):
        frame_observations = KLT_TRACKER.track_frame(rect_frame, depth_frame)
        track_history.append(frame_observations)
        result = RELATIVE_POSE_INITIALIZER.process_frame(
            frame_index=frame_idx,
            rectified_frame=rect_frame,
            depth_frame=depth_frame,
            track_observations=frame_observations,
        )
        if result is not None:
            sequence_results.append(result)

    tracks = dict(KLT_TRACKER.tracks)

    tracking_summary = summarise_tracking_results(sequence_sample, track_history, sequence_results)

    ba_result = run_fixed_lag_adjustment(
        bundle_adjuster=BUNDLE_ADJUSTER,
        rectified_frames=rectified_frames,
        depth_frames=depth_frames,
        track_history=track_history,
        tracks=tracks,
        sequence_results=sequence_results,
        sequence_sample=sequence_sample,
        use_imu=USE_IMU_FACTORS,
    )
    _log_bundle_adjuster_bias_diagnostics(ba_result, depth_label)

    ba_result_no_imu = None
    if USE_IMU_FACTORS:
        print("Skipping vision-only bundle adjustment; FixedLagBundleAdjuster currently requires IMU factors.")

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

    # rms_before = float(np.sqrt(np.mean(ba_result["reprojection_before"] ** 2)))
    # rms_after = float(np.sqrt(np.mean(ba_result["reprojection_after"] ** 2)))

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
        # "rms_before": rms_before,
        # "rms_after": rms_after,
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
    preprocessed = DATA_PIPELINE.prepare(seed=seed, max_depth=MAX_DEPTH)
    sequence = preprocessed.sequence
    dataset_label = preprocessed.label
    plot_imu_measurements(sequence, title=f"IMU data (sample {sample_idx})")
    inertial_poses = integrate_inertial_trajectory(sequence)
    rectified_frames = preprocessed.rectified_frames
    depth_variants: list[tuple[str, list[StereoDepthFrame]]] = [
        (variant.label, variant.frames) for variant in preprocessed.depth_variants
    ]
    if not depth_variants:
        raise RuntimeError("No depth variants produced by the data pipeline.")
    available_labels = ", ".join(label for label, _ in depth_variants)
    print(f"Depth variants available ({dataset_label}): {available_labels}")
    if DEPTH_MODE not in {label for label, _ in depth_variants}:
        raise RuntimeError(
            f"Depth variant '{DEPTH_MODE}' requested via LOCAL_VO_DEPTH_MODE but only "
            f"{available_labels or 'none'} are available."
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
    # rms_before = primary_result["rms_before"]
    # rms_after = primary_result["rms_after"]
    track_history = primary_result["track_history"]
    tracks = primary_result["tracks"]
    sequence_results = primary_result["sequence_results"]

    for result in variant_results:
        label = result["label"]
        result_ba = result["ba_result"]
        print(
            f"{'Bundle adjustment summary (with IMU)' if USE_IMU_FACTORS else 'Bundle adjustment summary'} "
            f"[{label}]"
        )
        print(f"  Frames optimised: {result_ba['frames_for_ba']}")
        print(f"  Landmarks optimised: {len(result_ba['landmark_original_indices'])}")
        # print(
        #     f"  Reprojection RMS: {result['rms_before']:.3f} px -> {result['rms_after']:.3f} px"
        # )
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
            # rms_before_no_imu = float(np.sqrt(np.mean(result_no_imu["reprojection_before"] ** 2)))
            # rms_after_no_imu = float(np.sqrt(np.mean(result_no_imu["reprojection_after"] ** 2)))
            print("Bundle adjustment summary (vision only)")
            print(f"  Frames optimised: {result_no_imu['frames_for_ba']}")
            print(f"  Landmarks optimised: {len(result_no_imu['landmark_original_indices'])}")
            # print(f"  Reprojection RMS: {rms_before_no_imu:.3f} px -> {rms_after_no_imu:.3f} px")
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
    # plot_track_length_histogram(tracks)
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
    # plot_pose_trajectories(
    #     ba_result["initial_pose_dict"],
    #     ba_result["optimized_pose_dict"],
    #     sequence,
    #     landmark_stride=5,
    #     inertial_poses=inertial_poses,
    #     optimized_pose_dict_no_imu=ba_result_no_imu["optimized_pose_dict"] if ba_result_no_imu else None,
    #     initial_label=primary_initial_label,
    #     optimized_label=primary_optimized_label,
    #     additional_pose_dicts=additional_pose_dicts,
    #     additional_initial_pose_dicts=additional_initial_pose_dicts,
    # )
    # plot_pose_error_norms_per_frame(
    #     ba_result["initial_pose_dict"],
    #     ba_result["optimized_pose_dict"],
    #     sequence,
    #     comparison_pose_dict=ba_result_no_imu["optimized_pose_dict"] if ba_result_no_imu else None,
    # )
    # plot_reprojection_error_histograms(
    #     ba_result["reprojection_before"],
    #     ba_result["reprojection_after"],
    # )
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

        save_variant_tum_files(
            sequence_label=dataset_label,
            sample_index=sample_idx,
            depth_label=result["label"],
            estimated_pose_dict=result["optimized_pose_dict"],
            sequence_sample=sequence,
        )

        # Pass the ground truth trajectory directly to the evaluation function.
        optimized_frame_indices = sorted(result["optimized_pose_dict"])
        est_traj_xyz = np.asarray([result["optimized_pose_dict"][idx].translation() for idx in optimized_frame_indices])
        est_traj_quat = [result["optimized_pose_dict"][idx].rotation().toQuaternion() for idx in optimized_frame_indices]
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
        err_xyzs.append((result["label"], err_xyz))

        print(f"Result: {result['label']}")
        print(f"GT total distance: {gt_traj_total_distance:.2f} m")
        print(f"ATE per meter travelled: {(ate / gt_traj_total_distance) * 100:.2f}%")

        if result['label'] == 'sgbm':
            rr.init("local_vo_bundle_adjustment")
            rr.spawn()

            est_poses = []
            gt_poses = []
            initial_est_poses = []
            initial_pose_dict = result.get("initial_pose_dict", {})
            num_frames = min(len(optimized_frame_indices), len(est_traj_aligned), len(gt_traj))

            # log map
            rr.set_time("frame", sequence=0)

            optimized_landmarks = result["ba_result"].get("optimized_landmarks")
            if optimized_landmarks:
                landmarks_arr = np.asarray(optimized_landmarks, dtype=np.float32)
                if landmarks_arr.size:
                    rr.log("optimized_landmarks", rr.Points3D(landmarks_arr, radii=0.01))

            # log poses trajectories
            for i in range(num_frames):
                rr.set_time("frame", sequence=i)
                frame_idx = optimized_frame_indices[i]

                # use the unaligned est_traj for now so the landmarks don't also need to be aligned
                est_pose = gtsam.Pose3(
                    gtsam.Rot3.Quaternion(w=est_traj[i, 3], x=est_traj[i, 4], y=est_traj[i, 5], z=est_traj[i, 6]),
                    gtsam.Point3(est_traj[i, 0:3])
                )

                est_poses.append(est_pose)
                rr_log_pose("est_pose", est_pose, depth_variants[1][1][i] if len(depth_variants) > 1 else depth_variants[0][1][i], camera_xyz=rr.ViewCoordinates.RIGHT_HAND_X_UP)

                gt_pose = gtsam.Pose3(
                    gtsam.Rot3.Quaternion(w=gt_traj[i, 6], x=gt_traj[i, 3], y=gt_traj[i, 4], z=gt_traj[i, 5]),
                    gtsam.Point3(gt_traj[i, 0:3])
                )
                gt_poses.append(gt_pose)
                # rr_log_pose("gt_pose", gt_pose, rectified_frames[0][1][i])

                rr_log_trajectory("est_trajectory", est_poses, color=(0, 0, 255), radii=0.006)
                rr_log_trajectory("gt_trajectory", gt_poses, color=(0, 255, 0), radii=0.006)
                initial_pose = initial_pose_dict.get(frame_idx)
                if initial_pose is not None:
                    initial_est_poses.append(initial_pose)
                    rr_log_trajectory("initial_est_trajectory", initial_est_poses, color=(255, 165, 0), radii=0.006)

    plot_translation_error_percentages(percentage_plot_series)

    for label, err_xyz in err_xyzs:
        plt.plot(range(len(err_xyz)), err_xyz, label=label)
    plt.legend()
    plt.show()


    all_tracking_results.append(sequence_results)
    all_ba_results.append(ba_result)

    # %%
