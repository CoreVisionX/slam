from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Tuple, Generic, TypeVar

import cv2
import gtsam
from registration.registration import RectifiedStereoFrame, StereoCalibration, StereoFrame, FramePairWithGroundTruth
import tartanair as ta
from cvx_utils import convert_coordinate_frame, se3_flattened_to_pose3
import numpy as np

# setup tartanair
tartanair_data_root = str(Path(__file__).parent / 'data')
ta.init(tartanair_data_root)

kitti_data_root = Path(__file__).parent / 'data' / 'ktti'

# converts the TartanAir coordinate frame to the CV coordinate frame
TA_TO_CV = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])


# https://tartanair.org/modalities.html
WIDTH = 640
HEIGHT = 640
GRAVITY_VECTOR_CV = np.array([0.0, -9.81, 0.0], dtype=np.float64)
tartanair_calib = StereoCalibration.create(
    K=np.array([
        [WIDTH / 2, 0.0, WIDTH / 2],
        [0.0, HEIGHT / 2, HEIGHT / 2],
        [0.0, 0.0, 1.0]
    ]),
    T=np.array([0.25, 0.0, 0.0]),
    R=np.eye(3),
    width=WIDTH, height=HEIGHT
)

@dataclass
class FrameImuMeasurements:
    frame_timestamp: float
    timestamps: np.ndarray
    dts: np.ndarray
    linear_accelerations: np.ndarray
    angular_velocities: np.ndarray
    world_velocity: np.ndarray | None = None
    body_velocity: np.ndarray | None = None

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64).reshape(-1)
        linear_acc = np.asarray(self.linear_accelerations, dtype=np.float64)
        angular_vel = np.asarray(self.angular_velocities, dtype=np.float64)

        if linear_acc.size == 0:
            linear_acc = np.zeros((0, 3), dtype=np.float64)
        else:
            linear_acc = linear_acc.reshape(-1, 3)

        if angular_vel.size == 0:
            angular_vel = np.zeros((0, 3), dtype=np.float64)
        else:
            angular_vel = angular_vel.reshape(-1, 3)

        if linear_acc.shape[0] != timestamps.size or angular_vel.shape[0] != timestamps.size:
            raise ValueError("IMU measurement counts must align between timestamps, accel, and gyro.")

        world_vel = self.world_velocity
        if world_vel is None:
            world_vel = np.zeros(3, dtype=np.float64)
        else:
            world_vel = np.asarray(world_vel, dtype=np.float64).reshape(3)

        body_vel = self.body_velocity
        if body_vel is None:
            body_vel = np.zeros(3, dtype=np.float64)
        else:
            body_vel = np.asarray(body_vel, dtype=np.float64).reshape(3)

        self.timestamps = timestamps
        self.linear_accelerations = linear_acc
        self.angular_velocities = angular_vel
        self.world_velocity = world_vel
        self.body_velocity = body_vel

    def __len__(self) -> int:
        return self.timestamps.size


def add_sch16t_noise(omega_true, acc_true, dt, seed=0, tau_bias=500.0):
    """
    omega_true: (N,3) rad/s   (body rates)
    acc_true:   (N,3) m/s^2   (specific force incl. gravity per your convention)
    dt:         float seconds (assumed constant; if not, pass a vector and adapt)
    tau_bias:   bias correlation time [s] for OU bias (200–1000 s reasonable)

    Returns: (omega_noisy, acc_noisy)
    """
    rng = np.random.default_rng(seed)
    N = omega_true.shape[0]

    # SCH16T typical (datasheet)
    gyro_nd = 0.0006 * np.pi / 180.0      # rad/s/√Hz
    accel_nd = 0.8e-3                    # m/s²/√Hz
    gyro_bias_inst = 0.5 * np.pi / 180.0 / 3600.0  # rad/s
    accel_bias_inst = 0.20e-3                    # m/s²

    # Discrete white-noise std (per sample)
    sg = gyro_nd / np.sqrt(dt)
    sa = accel_nd / np.sqrt(dt)

    # OU bias with steady-state std = bias-instability, corr time tau_bias
    phi = np.exp(-dt / tau_bias)
    s_g_u = gyro_bias_inst * np.sqrt(1 - phi**2)
    s_a_u = accel_bias_inst * np.sqrt(1 - phi**2)

    b_g = np.zeros(3)
    b_a = np.zeros(3)
    omega_noisy = np.empty_like(omega_true)
    acc_noisy = np.empty_like(acc_true)

    for k in range(N):
        b_g = phi * b_g + s_g_u * rng.standard_normal(3)
        b_a = phi * b_a + s_a_u * rng.standard_normal(3)
        omega_noisy[k] = omega_true[k] + b_g + sg * rng.standard_normal(3)
        acc_noisy[k] = acc_true[k] + b_a + sa * rng.standard_normal(3)
    return omega_noisy, acc_noisy


@dataclass
class TartanAirSequence:
    left_paths: list[Path]
    right_paths: list[Path]
    poses: list[gtsam.Pose3]
    frame_ids: list[int]
    frame_timestamps: np.ndarray | None = None
    imu_measurements: list[FrameImuMeasurements] | None = None
    left_depth_paths: list[Path] | None = None

    @property
    def size(self) -> int:
        return len(self.left_paths)


@dataclass
class KittiSequence:
    left_paths: list[Path]
    right_paths: list[Path]
    poses: list[gtsam.Pose3]
    frame_ids: list[int]
    calibration: StereoCalibration

    @property
    def size(self) -> int:
        return len(self.left_paths)


_tartanair_sequence_cache: Dict[Tuple[str, str, str], TartanAirSequence] = {}
_kitti_sequence_cache: Dict[str, KittiSequence] = {}

S = TypeVar("S")


@dataclass
class FrameSequenceWithGroundTruth(Generic[S]):
    frames: list[S]
    world_poses: list[gtsam.Pose3]
    frame_indices: list[int]
    frame_ids: list[int]
    imu_measurements: list[FrameImuMeasurements] | None = None
    full_imu_measurements: list[FrameImuMeasurements] | None = None
    frame_timestamps: np.ndarray | None = None
    ground_truth_depths: list[np.ndarray] | None = None

    @property
    def length(self) -> int:
        return len(self.frames)

    def relative_pose(self, first_idx: int, second_idx: int) -> gtsam.Pose3:
        if first_idx < 0 or second_idx < 0:
            raise ValueError("Frame indices must be non-negative.")
        if first_idx >= self.length or second_idx >= self.length:
            raise IndexError("Frame index out of range for this sequence.")
        return self.world_poses[first_idx].inverse() * self.world_poses[second_idx]

    def get_frame_imu(self, idx: int) -> FrameImuMeasurements | None:
        if idx < 0 or idx >= self.length:
            raise IndexError("Frame index out of range for this sequence.")
        if self.imu_measurements is None:
            return None
        return self.imu_measurements[idx]


def _parse_frame_id(path: Path, fallback: int) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            pass
    return fallback


def _load_array_from_disk(base_path: Path) -> np.ndarray:
    npy_path = base_path.with_suffix(".npy")
    txt_path = base_path.with_suffix(".txt")
    if npy_path.exists():
        return np.load(npy_path)
    if txt_path.exists():
        return np.loadtxt(txt_path)
    raise FileNotFoundError(f"Could not find array at {npy_path} or {txt_path}")


def _load_tartanair_imu_measurements(imu_dir: Path, num_frames: int) -> tuple[list[FrameImuMeasurements], np.ndarray]:
    required_files = ["acc", "gyro", "imu_time", "cam_time", "vel_global", "vel_body"]
    arrays: dict[str, np.ndarray] = {}
    for name in required_files:
        arrays[name] = _load_array_from_disk(imu_dir / name).astype(np.float64, copy=False)

    acc = arrays["acc"]
    gyro = arrays["gyro"]
    imu_time = arrays["imu_time"].reshape(-1)
    cam_time = arrays["cam_time"].reshape(-1)
    vel_global = arrays["vel_global"]
    vel_body = arrays["vel_body"]

    imu_dts = np.diff(imu_time)
    # TODO: make sure 1: is right here instead of -1:
    imu_time = imu_time[1:]
    acc = acc[1:]
    gyro = gyro[1:]
    vel_global = vel_global[1:]
    vel_body = vel_body[1:]

    if cam_time.size != num_frames:
        raise ValueError(
            f"Camera timestamp count ({cam_time.size}) does not match number of frames ({num_frames})."
        )
    if acc.shape[0] != imu_time.size or gyro.shape[0] != imu_time.size:
        raise ValueError(
            "IMU accel/gyro samples must align with imu_time entries."
        )

    acc = (TA_TO_CV @ acc.T).T
    gyro = (TA_TO_CV @ gyro.T).T
    vel_global = (TA_TO_CV @ vel_global.T).T
    vel_body = (TA_TO_CV @ vel_body.T).T

    measurements: list[FrameImuMeasurements] = []
    for frame_idx, frame_timestamp in enumerate(cam_time):
        if frame_idx < 1:
            # add an empty batch for the first frame
            measurements.append(
                FrameImuMeasurements(
                    frame_timestamp=float(frame_timestamp),
                    timestamps=np.array([], dtype=np.float64),
                    dts=np.array([], dtype=np.float64),
                    linear_accelerations=np.array([], dtype=np.float64),
                    angular_velocities=np.array([], dtype=np.float64),
                )
            )

            continue

        prev_frame_timestamp = cam_time[frame_idx - 1]
        frame_timestamp = cam_time[frame_idx]
        mask = (np.round(imu_time, 3) > np.round(prev_frame_timestamp, 2)) & (np.round(imu_time, 3) <= np.round(frame_timestamp, 2))

        timestamps = imu_time[mask]
        if frame_idx != num_frames - 1:
            assert len(timestamps) == 10, f"Expected 10 IMU samples for frame {frame_idx}, got {len(timestamps)}: {timestamps} for {prev_frame_timestamp} to {frame_timestamp}"
        else:
            assert len(timestamps) == 9, f"Expected 9 IMU samples for last frame {frame_idx}, got {len(timestamps)}: {timestamps} for {prev_frame_timestamp} to {frame_timestamp}"

        dts = imu_dts[mask]
        lin_acc = acc[mask]
        ang_vel = gyro[mask]
        frame_vel_world = vel_global[mask][-1] # TODO: store whole batch of velocities? this should be closest to the camera right now?
        measurements.append(
            FrameImuMeasurements(
                frame_timestamp=float(frame_timestamp),
                timestamps=timestamps,
                dts=dts,
                linear_accelerations=lin_acc,
                angular_velocities=ang_vel,
                world_velocity=frame_vel_world,
            )
        )

    assert len(measurements) == num_frames
    print(f'Loaded {len(measurements)} IMU measurements')

    return measurements, cam_time


def _load_tartanair_sequence(env: str, difficulty: str, traj: str) -> TartanAirSequence:
    key = (env, difficulty, traj)
    if key in _tartanair_sequence_cache:
        return _tartanair_sequence_cache[key]

    traj_root = Path(tartanair_data_root) / env / f"Data_{difficulty}" / traj
    left_dir = traj_root / "image_lcam_front"
    right_dir = traj_root / "image_rcam_front"
    pose_path = traj_root / "pose_lcam_front.txt"

    if not left_dir.exists() or not right_dir.exists():
        raise FileNotFoundError(f"Missing image directory for {env}/{difficulty}/{traj}")
    if not pose_path.exists():
        raise FileNotFoundError(f"Missing pose file at {pose_path}")

    left_paths = sorted(left_dir.glob("*.png"))
    right_paths = sorted(right_dir.glob("*.png"))

    if len(left_paths) != len(right_paths):
        raise ValueError(f"Left/right image count mismatch for {env}/{difficulty}/{traj}")

    depth_dir = traj_root / "depth_lcam_front"
    left_depth_paths: list[Path] | None = None
    if depth_dir.exists():
        candidate_paths: list[Path] = []
        missing_depth = False
        for left_path in left_paths:
            depth_path = depth_dir / f"{left_path.stem}_depth{left_path.suffix}"
            if not depth_path.exists():
                missing_depth = True
                break
            candidate_paths.append(depth_path)
        if not missing_depth:
            left_depth_paths = candidate_paths

    pose_array = np.loadtxt(pose_path)
    if pose_array.ndim == 1:
        pose_array = pose_array.reshape(1, -1)

    poses: list[gtsam.Pose3] = []
    for row in pose_array:
        if row.size == 12:
            pose = se3_flattened_to_pose3(row.astype(np.float64))
        elif row.size == 7:
            tx, ty, tz, qx, qy, qz, qw = row.astype(np.float64)
            rotation = gtsam.Rot3.Quaternion(qw, qx, qy, qz)
            pose = gtsam.Pose3(rotation, gtsam.Point3(tx, ty, tz))
        else:
            raise ValueError(f"Unsupported pose format with {row.size} values.")
        pose = convert_coordinate_frame(pose, TA_TO_CV)
        poses.append(pose)

    if len(poses) != len(left_paths):
        raise ValueError("Pose count does not match number of images.")

    frame_ids = [_parse_frame_id(path, idx) for idx, path in enumerate(left_paths)]
    imu_dir = traj_root / "imu"
    imu_measurements: list[FrameImuMeasurements] | None = None
    frame_timestamps: np.ndarray | None = None
    if imu_dir.exists():
        imu_measurements, frame_timestamps = _load_tartanair_imu_measurements(
            imu_dir, len(left_paths)
        )
        print('imu measurements loaded', len(imu_measurements))

    sequence = TartanAirSequence(
        left_paths=left_paths,
        right_paths=right_paths,
        poses=poses,
        frame_ids=frame_ids,
        frame_timestamps=frame_timestamps,
        imu_measurements=imu_measurements,
        left_depth_paths=left_depth_paths,
    )
    _tartanair_sequence_cache[key] = sequence
    return sequence

def _load_stereo_frame(sequence: TartanAirSequence, index: int) -> StereoFrame:
    left_path = sequence.left_paths[index]
    right_path = sequence.right_paths[index]

    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)

    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

    if left is None or right is None:
        raise ValueError(f"Failed to load images for index {index} at {left_path} / {right_path}")

    if left.shape[0] != HEIGHT or left.shape[1] != WIDTH:
        left = cv2.resize(left, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    if right.shape[0] != HEIGHT or right.shape[1] != WIDTH:
        right = cv2.resize(right, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

    return StereoFrame(
        left=left,
        right=right,
        calibration=tartanair_calib
    )


def _read_tartanair_depth_map(depth_path: Path) -> np.ndarray:
    suffix = depth_path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(depth_path)
    else:
        depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise FileNotFoundError(f"Failed to load depth map at {depth_path}")
        depth_image = np.ascontiguousarray(depth_image)
        if depth_image.ndim == 2 and depth_image.dtype != np.uint8:
            depth = depth_image.astype(np.float32, copy=False)
        elif depth_image.ndim == 2:
            depth = depth_image.astype(np.float32, copy=False)
        elif depth_image.ndim == 3 and depth_image.shape[2] >= 4:
            depth = depth_image.view("<f4")
            depth = np.squeeze(depth, axis=-1)
        else:
            raise ValueError(f"Unsupported depth image format at {depth_path}")

    if depth.ndim != 2:
        raise ValueError(f"Depth map at {depth_path} must be 2D.")

    if depth.shape != (HEIGHT, WIDTH):
        depth = cv2.resize(depth, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

    return depth.astype(np.float32, copy=False)


def _normalize_kitti_sequence_id(sequence_id: str | int) -> str:
    seq_str = str(sequence_id).strip()
    if not seq_str:
        raise ValueError("sequence_id cannot be empty.")
    if seq_str.isdigit() and len(seq_str) < 2:
        seq_str = seq_str.zfill(2)
    return seq_str


def _projection_matrix_to_pose(P: np.ndarray) -> np.ndarray:
    """
    Convert a 3x4 projection matrix into a 4x4 pose transforming camera coordinates to world coordinates.
    """
    if P.shape != (3, 4):
        raise ValueError("Projection matrix must be 3x4.")

    K = P[:, :3]
    K_inv = np.linalg.inv(K)
    Rt = K_inv @ P
    R_world_to_cam = Rt[:, :3]
    t_world_to_cam = Rt[:, 3]

    R_cam_to_world = R_world_to_cam.T
    t_cam_to_world = -R_cam_to_world @ t_world_to_cam

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R_cam_to_world
    pose[:3, 3] = t_cam_to_world
    return pose


def _identity_rectification_maps(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build rectification maps that leave the image unchanged (used when data is already rectified).
    """
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    map_x, map_y = np.meshgrid(xs, ys)
    return map_x, map_y


def _rectified_q_matrix(
    K_left_rect: np.ndarray,
    K_right_rect: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """
    Construct the reprojection matrix for a rectified stereo pair using standard
    OpenCV conventions (see reprojectImageTo3D documentation).
    """
    Tx = float(np.asarray(T, dtype=np.float64).ravel()[0])
    if np.isclose(Tx, 0.0):
        raise ValueError("KITTI baseline is zero; cannot build reprojection matrix.")

    cx_left = float(K_left_rect[0, 2])
    cy_left = float(K_left_rect[1, 2])
    cx_right = float(K_right_rect[0, 2])
    fx = float(K_left_rect[0, 0])

    Q = np.array([
        [1.0, 0.0, 0.0, -cx_left],
        [0.0, 1.0, 0.0, -cy_left],
        [0.0, 0.0, 0.0, fx],
        [0.0, 0.0, -1.0 / Tx, (cx_left - cx_right) / Tx],
    ], dtype=np.float64)
    return Q


def _load_kitti_sequence(sequence_id: str | int) -> KittiSequence:
    seq_id = _normalize_kitti_sequence_id(sequence_id)
    if seq_id in _kitti_sequence_cache:
        return _kitti_sequence_cache[seq_id]

    sequence_root = kitti_data_root / 'sequences' / seq_id
    poses_root = kitti_data_root / 'poses'

    if not sequence_root.exists():
        raise FileNotFoundError(f"Missing KITTI sequence at {sequence_root}")

    left_dir = sequence_root / "image_2"
    right_dir = sequence_root / "image_3"
    calib_path = sequence_root / "calib.txt"
    pose_path = poses_root / f"{seq_id}.txt"

    if not left_dir.exists() or not right_dir.exists():
        raise FileNotFoundError(f"Missing stereo image directories for sequence {seq_id}")
    if not calib_path.exists():
        raise FileNotFoundError(f"Missing calibration file at {calib_path}")
    if not pose_path.exists():
        raise FileNotFoundError(f"Missing pose file at {pose_path}")

    left_paths = sorted(left_dir.glob("*.png"))
    right_paths = sorted(right_dir.glob("*.png"))

    if not left_paths or not right_paths:
        raise ValueError(f"No images found for KITTI sequence {seq_id}")
    if len(left_paths) != len(right_paths):
        raise ValueError(f"Left/right image count mismatch for KITTI sequence {seq_id}")

    sample_left = cv2.imread(str(left_paths[0]), cv2.IMREAD_COLOR)
    sample_right = cv2.imread(str(right_paths[0]), cv2.IMREAD_COLOR)

    if sample_left is None or sample_right is None:
        raise ValueError(f"Failed to load sample images for KITTI sequence {seq_id}")
    if sample_left.shape != sample_right.shape:
        raise ValueError(f"Left/right image shapes differ for KITTI sequence {seq_id}")

    height, width = sample_left.shape[:2]

    projection_mats: Dict[str, np.ndarray] = {}
    for line in calib_path.read_text().strip().splitlines():
        if not line:
            continue
        key, values = line.split(":", 1)
        data = np.fromstring(values, sep=" ", dtype=np.float64)
        if data.size != 12:
            raise ValueError(f"Unexpected calibration format in {calib_path}")
        projection_mats[key.strip()] = data.reshape(3, 4)

    for key in ("P0", "P2", "P3"):
        if key not in projection_mats:
            raise ValueError(f"Missing projection matrix '{key}' in {calib_path}")

    pose_cam0 = _projection_matrix_to_pose(projection_mats["P0"])
    pose_left = _projection_matrix_to_pose(projection_mats["P2"])
    pose_right = _projection_matrix_to_pose(projection_mats["P3"])

    left_to_cam0 = np.linalg.inv(pose_cam0) @ pose_left
    left_to_right = np.linalg.inv(pose_left) @ pose_right

    K_left = projection_mats["P2"][:, :3].copy()
    K_right = projection_mats["P3"][:, :3].copy()
    K_left /= K_left[2, 2]
    K_right /= K_right[2, 2]

    D_left = np.zeros((5, 1), dtype=np.float64)
    D_right = np.zeros((5, 1), dtype=np.float64)

    R = left_to_right[:3, :3]
    T = left_to_right[:3, 3].reshape(3, 1)

    K_left_rect = K_left.copy()
    K_right_rect = K_right.copy()
    map_left_x, map_left_y = _identity_rectification_maps(width, height)
    map_right_x, map_right_y = _identity_rectification_maps(width, height)
    Q = _rectified_q_matrix(K_left_rect, K_right_rect, T)

    calibration = StereoCalibration(
        K_left=K_left,
        K_right=K_right,
        K_left_rect=K_left_rect,
        K_right_rect=K_right_rect,
        D_left=D_left,
        D_right=D_right,
        R=R,
        T=T,
        R_left=np.eye(3, dtype=np.float64),
        R_right=np.eye(3, dtype=np.float64),
        P_left=projection_mats["P2"].copy(),
        P_right=projection_mats["P3"].copy(),
        Q=Q,
        map_left_x=map_left_x,
        map_left_y=map_left_y,
        map_right_x=map_right_x,
        map_right_y=map_right_y,
        width=width,
        height=height,
    )

    pose_array = np.loadtxt(pose_path, dtype=np.float64)
    pose_array = pose_array.reshape(-1, 12)
    pose_matrices = pose_array.reshape(-1, 3, 4)

    if pose_matrices.shape[0] < len(left_paths):
        raise ValueError(
            f"Pose count ({pose_matrices.shape[0]}) shorter than image count "
            f"({len(left_paths)}) for sequence {seq_id}"
        )

    poses: list[gtsam.Pose3] = []
    for matrix in pose_matrices[:len(left_paths)]:
        T_w_cam0 = np.eye(4, dtype=np.float64)
        T_w_cam0[:3, :3] = matrix[:, :3]
        T_w_cam0[:3, 3] = matrix[:, 3]

        T_w_cam_left = T_w_cam0 @ left_to_cam0

        pose = gtsam.Pose3(
            gtsam.Rot3(T_w_cam_left[:3, :3]),
            gtsam.Point3(T_w_cam_left[:3, 3]),
        )
        poses.append(pose)

    frame_ids = [_parse_frame_id(path, idx) for idx, path in enumerate(left_paths)]

    sequence = KittiSequence(
        left_paths=left_paths,
        right_paths=right_paths,
        poses=poses,
        frame_ids=frame_ids,
        calibration=calibration,
    )
    _kitti_sequence_cache[seq_id] = sequence
    return sequence


def _load_kitti_frame(sequence: KittiSequence, index: int) -> StereoFrame:
    left_path = sequence.left_paths[index]
    right_path = sequence.right_paths[index]

    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)

    if left is None or right is None:
        raise ValueError(f"Failed to load images for index {index} at {left_path} / {right_path}")

    left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

    return StereoFrame(
        left=left,
        right=right,
        calibration=sequence.calibration,
    )


def load_kitti_sequence_segment(
    sequence_id: str | int = "00",
    sequence_length: int = 4,
    seed: int = 0,
    sampling_mode: str = "contiguous",
    min_stride: int = 1,
    max_stride: int | None = None,
) -> FrameSequenceWithGroundTruth[StereoFrame]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    if min_stride < 1:
        raise ValueError("min_stride must be >= 1.")
    if max_stride is None:
        max_stride = min_stride
    if max_stride < min_stride:
        raise ValueError("max_stride must be >= min_stride.")

    sequence = _load_kitti_sequence(sequence_id)
    if sequence.size < sequence_length:
        raise ValueError(
            f"Sequence too short ({sequence.size}) for requested length {sequence_length}."
        )

    rng = random.Random(seed)
    sampling_mode = sampling_mode.lower()

    if sampling_mode == "contiguous":
        max_start = sequence.size - sequence_length
        start_idx = rng.randint(0, max_start)
        sampled_indices = list(range(start_idx, start_idx + sequence_length))
    elif sampling_mode == "stride":
        max_attempts = min(1024, sequence.size * 8)
        sampled_indices: list[int] = []
        for _ in range(max_attempts):
            start_idx = rng.randint(0, sequence.size - 1)
            current_idx = start_idx
            candidate = [current_idx]
            for _ in range(sequence_length - 1):
                stride = rng.randint(min_stride, max_stride)
                current_idx += stride
                if current_idx >= sequence.size:
                    break
                candidate.append(current_idx)
            if len(candidate) == sequence_length:
                sampled_indices = candidate
                break
        if len(sampled_indices) != sequence_length:
            raise RuntimeError(
                "Could not sample a sequence satisfying the requested stride bounds."
            )
    else:
        raise ValueError(
            f"Unsupported sampling_mode '{sampling_mode}'. Expected 'contiguous' or 'stride'."
        )

    frames = [_load_kitti_frame(sequence, idx) for idx in sampled_indices]
    poses = [sequence.poses[idx] for idx in sampled_indices]
    frame_ids = [sequence.frame_ids[idx] for idx in sampled_indices]

    return FrameSequenceWithGroundTruth[StereoFrame](
        frames=frames,
        world_poses=poses,
        frame_indices=sampled_indices,
        frame_ids=frame_ids,
    )


def get_kitti_iterator_with_odometry(
    sequence_id: str | int = "00",
    rotation_noise_sigmas: np.ndarray | None = None,
    translation_noise_sigmas: np.ndarray | None = None,
    include_ground_truth: bool = False,
):
    sequence = _load_kitti_sequence(sequence_id)

    if rotation_noise_sigmas is None:
        rotation_noise_sigmas = np.zeros(3, dtype=np.float64)
    else:
        rotation_noise_sigmas = np.asarray(rotation_noise_sigmas, dtype=np.float64).reshape(3)

    if translation_noise_sigmas is None:
        translation_noise_sigmas = np.zeros(3, dtype=np.float64)
    else:
        translation_noise_sigmas = np.asarray(translation_noise_sigmas, dtype=np.float64).reshape(3)

    world_to_prev_robot: gtsam.Pose3 | None = None
    world_to_first_robot: gtsam.Pose3 | None = None

    for idx in range(sequence.size):
        world_to_robot = sequence.poses[idx]

        if world_to_prev_robot is None or world_to_first_robot is None:
            world_to_prev_robot = world_to_robot
            world_to_first_robot = world_to_robot
            continue

        prev_robot_to_robot = world_to_prev_robot.inverse() * world_to_robot

        rotation_noise = gtsam.Rot3.Expmap(np.random.normal(0.0, rotation_noise_sigmas))
        translation_noise = gtsam.Point3(np.random.normal(0.0, translation_noise_sigmas))
        prev_robot_to_robot_noise = gtsam.Pose3(rotation_noise, translation_noise)
        noisy_prev_robot_to_robot = prev_robot_to_robot.compose(prev_robot_to_robot_noise)

        first_robot_to_robot = world_to_first_robot.inverse() * world_to_robot
        world_to_prev_robot = world_to_robot

        frame = _load_kitti_frame(sequence, idx)

        if include_ground_truth:
            yield frame, noisy_prev_robot_to_robot, first_robot_to_robot
        else:
            yield frame, noisy_prev_robot_to_robot


def get_kitti_calibration(sequence_id: str | int = "00") -> StereoCalibration:
    """
    Retrieve the stereo calibration for a KITTI sequence.
    """
    return _load_kitti_sequence(sequence_id).calibration



def _apply_sch16t_noise_to_measurements(
    measurements: list[FrameImuMeasurements],
    seed: int,
    tau_bias: float,
) -> list[FrameImuMeasurements]:
    """
    Clone IMU measurements while injecting SCH16T noise per timeseries.
    """
    if not measurements:
        return []

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=len(measurements), dtype=np.int64)

    noisy_measurements: list[FrameImuMeasurements] = []
    for meas, meas_seed in zip(measurements, seeds):
        timestamps = meas.timestamps.copy()
        dts = meas.dts.copy()
        omega_true = meas.angular_velocities
        acc_true = meas.linear_accelerations

        dt = 0.0
        if timestamps.size >= 2:
            dt = float(np.mean(np.diff(timestamps)))

        if dt > 0.0:
            omega_noisy, acc_noisy = add_sch16t_noise(
                omega_true,
                acc_true,
                dt,
                seed=int(meas_seed),
                tau_bias=float(tau_bias),
            )
        else:
            omega_noisy = omega_true.copy()
            acc_noisy = acc_true.copy()

        noisy_measurements.append(
            FrameImuMeasurements(
                frame_timestamp=meas.frame_timestamp,
                timestamps=timestamps,
                dts=dts,
                linear_accelerations=acc_noisy,
                angular_velocities=omega_noisy,
                world_velocity=meas.world_velocity.copy(),
                body_velocity=meas.body_velocity.copy(),
            )
        )

    print("Returning noisy measurements")
    return noisy_measurements


def load_tartanair_sequence_segment(
    env: str = "ArchVizTinyHouseDay",
    difficulty: str = "easy",
    traj: str = "P000",
    sequence_length: int = 4,
    seed: int = 0,
    sampling_mode: str = "contiguous",
    min_stride: int = 1,
    max_stride: int | None = None,
    load_ground_truth_depth: bool = False,
    add_imu_noise: bool = False,
    imu_noise_seed: int = 0,
    imu_noise_tau_bias: float = 500.0,
) -> FrameSequenceWithGroundTruth[StereoFrame]:
    """
    Sample a sequence of stereo frames with associated ground-truth poses.

    Parameters
    ----------
    sequence_length:
        Number of frames to return. Must be >= 2.
    sampling_mode:
        "contiguous" samples a contiguous block of frames.
        "stride" samples frames with random strides between min_stride and max_stride (inclusive).
    min_stride, max_stride:
        Bounds on the stride when sampling in "stride" mode. The stride applies between consecutive
        frames within the sequence. Defaults to 1 (contiguous sampling).
    load_ground_truth_depth:
        When True, load and return the dataset-provided left camera depth maps alongside the images.
    add_imu_noise:
        When True, inject SCH16T-style noise into the frame IMU data returned by this helper.
        Has no effect if the sequence has no IMU data.
    imu_noise_seed:
        Base RNG seed used when sampling noise for each frame chunk.
    imu_noise_tau_bias:
        Bias correlation time (seconds) passed to the SCH16T noise model.
    """

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    if min_stride < 1:
        raise ValueError("min_stride must be >= 1.")
    if max_stride is None:
        max_stride = min_stride
    if max_stride < min_stride:
        raise ValueError("max_stride must be >= min_stride.")

    sequence = _load_tartanair_sequence(env, difficulty, traj)
    if sequence.size < sequence_length:
        raise ValueError(
            f"Sequence too short ({sequence.size}) for requested length {sequence_length}."
        )

    rng = random.Random(seed)
    sampling_mode = sampling_mode.lower()

    if sampling_mode == "contiguous":
        max_start = sequence.size - sequence_length
        start_idx = rng.randint(0, max_start)
        sampled_indices = list(range(start_idx, start_idx + sequence_length))
    elif sampling_mode == "stride":
        max_attempts = min(1024, sequence.size * 8)
        sampled_indices = []
        for _ in range(max_attempts):
            start_idx = rng.randint(0, sequence.size - 1)
            current_idx = start_idx
            candidate = [current_idx]
            for _ in range(sequence_length - 1):
                stride = rng.randint(min_stride, max_stride)
                current_idx += stride
                if current_idx >= sequence.size:
                    break
                candidate.append(current_idx)
            if len(candidate) == sequence_length:
                sampled_indices = candidate
                break
        if len(sampled_indices) != sequence_length:
            raise RuntimeError(
                "Could not sample a sequence satisfying the requested stride bounds."
            )
    else:
        raise ValueError(
            f"Unsupported sampling_mode '{sampling_mode}'. Expected 'contiguous' or 'stride'."
        )

    frames = [_load_stereo_frame(sequence, idx) for idx in sampled_indices]
    poses = [sequence.poses[idx] for idx in sampled_indices]
    frame_ids = [sequence.frame_ids[idx] for idx in sampled_indices]
    imu_measurements = (
        [sequence.imu_measurements[idx] for idx in sampled_indices]
        if sequence.imu_measurements is not None
        else None
    )
    if imu_measurements is not None and add_imu_noise:
        imu_measurements = _apply_sch16t_noise_to_measurements(
            imu_measurements,
            seed=imu_noise_seed,
            tau_bias=imu_noise_tau_bias,
        )
    frame_timestamps = (
        np.asarray(sequence.frame_timestamps, dtype=np.float64)
        if sequence.frame_timestamps is not None
        else None
    )
    ground_truth_depths = None
    if load_ground_truth_depth:
        if sequence.left_depth_paths is None:
            raise FileNotFoundError(
                f"Ground-truth depth unavailable for {env}/{difficulty}/{traj}."
            )
        ground_truth_depths = [
            _read_tartanair_depth_map(sequence.left_depth_paths[idx])
            for idx in sampled_indices
        ]

    return FrameSequenceWithGroundTruth[StereoFrame](
        frames=frames,
        world_poses=poses,
        frame_indices=sampled_indices,
        frame_ids=frame_ids,
        imu_measurements=imu_measurements,
        full_imu_measurements=sequence.imu_measurements,
        frame_timestamps=frame_timestamps,
        ground_truth_depths=ground_truth_depths,
    )


def get_tartanair_iterator_with_odometry(env="ArchVizTinyHouseDay", difficulty="easy", traj="P000", rotation_noise_sigmas=np.array([np.deg2rad(0.2), np.deg2rad(0.2), np.deg2rad(0.2)]), translation_noise_sigmas=np.array([0.005, 0.005, 0.005]), include_ground_truth=False):
    ta_iterator = ta.iterator(
        env=[env],
        difficulty=difficulty,
        trajectory_id=traj,
        modality=["image"],
        camera_name=["lcam_front", "rcam_front"],
    )

    world_to_prev_robot = None
    world_to_first_robot = None

    for sample in ta_iterator:
        world_to_robot = se3_flattened_to_pose3(sample['lcam_front']['pose'])
        world_to_robot = convert_coordinate_frame(world_to_robot, TA_TO_CV)

        if world_to_prev_robot is None or world_to_first_robot is None:
            world_to_prev_robot = world_to_robot
            world_to_first_robot = world_to_robot
            continue # skip since we can't provide a relative pose having seen only a single pose

        prev_robot_to_robot = world_to_prev_robot.inverse() * world_to_robot

        # add noise to the relative odometry measurement
        rotation_noise = gtsam.Rot3.Expmap(np.random.normal(0, rotation_noise_sigmas))
        translation_noise = gtsam.Point3(np.random.normal(0, translation_noise_sigmas))
        prev_robot_to_robot_noise = gtsam.Pose3(rotation_noise, translation_noise)
        noisy_prev_robot_to_robot = prev_robot_to_robot.compose(prev_robot_to_robot_noise) # should be equivalent to prev_robot_to_robot_noise * prev_robot_to_robot?

        # prepare ground truth pose
        first_robot_to_robot = world_to_first_robot.inverse() * world_to_robot

        # update previous world to robot pose
        world_to_prev_robot = world_to_robot

        # prepare frame
        frame =StereoFrame(
            left=cv2.resize(sample['lcam_front']['image'], (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR) if sample['lcam_front']['image'].shape[0] != HEIGHT else sample['lcam_front']['image'],
            right=cv2.resize(sample['rcam_front']['image'], (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR) if sample['rcam_front']['image'].shape[0] != HEIGHT else sample['rcam_front']['image'],
            calibration=tartanair_calib
        )

        if include_ground_truth:
            yield frame, noisy_prev_robot_to_robot, first_robot_to_robot
        else:
            yield frame, noisy_prev_robot_to_robot


def load_tartanair_pair(
    env="ArchVizTinyHouseDay",
    difficulty="easy",
    traj="P000",
    max_dist=1.0,
    max_degs=20.0,
    seed=0,
    sampling_mode: str = "motion",
    min_index_offset: int = 1,
    max_index_offset: int | None = None,
):
    """
    Load a pair of sensor data samples from the TartanAir dataset.

    The default "motion" sampling selects the pair that best matches a random motion magnitude
    drawn within the distance/rotation bounds. For a more uniform coverage across the trajectory,
    set sampling_mode="index_offset" to sample pairs by their frame index offset instead.
    """

    sequence = _load_tartanair_sequence(env, difficulty, traj)
    if sequence.size < 2:
        raise ValueError("Sequence must contain at least two frames.")

    rng = random.Random(seed)
    sampling_mode = sampling_mode.lower()

    if sampling_mode == "index_offset":
        if max_index_offset is None:
            max_index_offset = sequence.size - 1
        if min_index_offset < 1:
            raise ValueError("min_index_offset must be >= 1.")
        if max_index_offset < min_index_offset:
            raise ValueError("max_index_offset must be >= min_index_offset.")

        max_valid_offset = min(max_index_offset, sequence.size - 1)
        if max_valid_offset < min_index_offset:
            raise ValueError("Sequence too short for the requested index offset range.")

        offset = rng.randint(min_index_offset, max_valid_offset)
        last_start = sequence.size - 1 - offset
        first_idx = rng.randint(0, last_start)
        second_idx = first_idx + offset

        first_pose = sequence.poses[first_idx]
        second_pose = sequence.poses[second_idx]
        relative = first_pose.inverse() * second_pose
    elif sampling_mode == "motion":
        target_distance = rng.uniform(0.0, max_dist)
        target_angle_deg = rng.uniform(0.0, max_degs)
        max_attempts = min(1000, sequence.size * 10)

        best_candidate = None
        best_score = float("inf")

        for _ in range(max_attempts):
            first_idx = rng.randrange(0, sequence.size - 1)
            second_idx = rng.randrange(first_idx + 1, sequence.size)

            first_pose = sequence.poses[first_idx]
            second_pose = sequence.poses[second_idx]
            relative = first_pose.inverse() * second_pose

            distance = np.linalg.norm(relative.translation())
            angle_rad = np.linalg.norm(gtsam.Rot3.Logmap(relative.rotation()))
            angle_deg = np.rad2deg(angle_rad)

            if distance > max_dist or angle_deg > max_degs:
                continue

            score = abs(distance - target_distance) + abs(angle_deg - target_angle_deg)
            if score < best_score:
                best_candidate = (first_idx, second_idx, first_pose, second_pose, relative)
                best_score = score
                if score < 1e-2:
                    break

        if best_candidate is None:
            # Fallback: scan forward from a random index to find the first valid pair.
            first_idx = rng.randrange(0, sequence.size - 1)
            first_pose = sequence.poses[first_idx]
            for second_idx in range(first_idx + 1, sequence.size):
                second_pose = sequence.poses[second_idx]
                relative = first_pose.inverse() * second_pose
                distance = np.linalg.norm(relative.translation())
                angle_rad = np.linalg.norm(gtsam.Rot3.Logmap(relative.rotation()))
                angle_deg = np.rad2deg(angle_rad)
                if distance <= max_dist and angle_deg <= max_degs:
                    best_candidate = (first_idx, second_idx, first_pose, second_pose, relative)
                    break

        if best_candidate is None:
            raise RuntimeError("Could not find a frame pair satisfying the requested constraints.")

        first_idx, second_idx, first_pose, second_pose, relative = best_candidate
    else:
        raise ValueError(f"Unsupported sampling_mode '{sampling_mode}'. Expected 'motion' or 'index_offset'.")

    first_frame = _load_stereo_frame(sequence, first_idx)
    second_frame = _load_stereo_frame(sequence, second_idx)

    first_frame_id = sequence.frame_ids[first_idx]
    second_frame_id = sequence.frame_ids[second_idx]

    pair = FramePairWithGroundTruth[StereoFrame](
        first=first_frame,
        second=second_frame,
        first_T_second=relative
    )

    pair.first_index = first_idx  # type: ignore[attr-defined]
    pair.second_index = second_idx  # type: ignore[attr-defined]
    pair.first_frame_id = first_frame_id  # type: ignore[attr-defined]
    pair.second_frame_id = second_frame_id  # type: ignore[attr-defined]
    pair.frame_indices = (first_idx, second_idx)  # type: ignore[attr-defined]
    pair.frame_ids = (first_frame_id, second_frame_id)  # type: ignore[attr-defined]

    return pair

def print_pose_error(*, estimated_pose: gtsam.Pose3 | None = None, ground_truth_pose: gtsam.Pose3 | None = None):
    if estimated_pose is not None and ground_truth_pose is not None:
        pose_error = estimated_pose * ground_truth_pose.inverse()
        
        print('--------------------------------')
        print(f'Estimated translation: ({np.linalg.norm(estimated_pose.translation()):.2f} m) [{estimated_pose.translation()[0]:.2f}, {estimated_pose.translation()[1]:.2f}, {estimated_pose.translation()[2]:.2f}]')
        print(f'Estimated rotation: ({np.linalg.norm(np.rad2deg(estimated_pose.rotation().ypr())):.2f} deg) [{np.rad2deg(estimated_pose.rotation().ypr()[0]):.2f}, {np.rad2deg(estimated_pose.rotation().ypr()[1]):.2f}, {np.rad2deg(estimated_pose.rotation().ypr()[2]):.2f}]')
        print('--------------------------------')
        print(f'Ground truth translation: ({np.linalg.norm(ground_truth_pose.translation()):.2f} m) [{ground_truth_pose.translation()[0]:.2f}, {ground_truth_pose.translation()[1]:.2f}, {ground_truth_pose.translation()[2]:.2f}]')
        print(f'Ground truth rotation: ({np.linalg.norm(np.rad2deg(ground_truth_pose.rotation().ypr())):.2f} deg) [{np.rad2deg(ground_truth_pose.rotation().ypr()[0]):.2f}, {np.rad2deg(ground_truth_pose.rotation().ypr()[1]):.2f}, {np.rad2deg(ground_truth_pose.rotation().ypr()[2]):.2f}]')
        print('--------------------------------')
        print(f'Pose error translation: ({np.linalg.norm(pose_error.translation()):.2f} m) [{pose_error.translation()[0]:.2f}, {pose_error.translation()[1]:.2f}, {pose_error.translation()[2]:.2f}]')
        print(f'Pose error rotation: ({np.linalg.norm(np.rad2deg(pose_error.rotation().ypr())):.2f} deg) [{np.rad2deg(pose_error.rotation().ypr()[0]):.2f}, {np.rad2deg(pose_error.rotation().ypr()[1]):.2f}, {np.rad2deg(pose_error.rotation().ypr()[2]):.2f}]')
        print('--------------------------------')
    elif estimated_pose is not None:
        print('--------------------------------')
        print(f'Estimated translation: ({np.linalg.norm(estimated_pose.translation()):.2f} m) [{estimated_pose.translation()[0]:.2f}, {estimated_pose.translation()[1]:.2f}, {estimated_pose.translation()[2]:.2f}]')
        print(f'Estimated rotation: ({np.linalg.norm(np.rad2deg(estimated_pose.rotation().ypr())):.2f} deg) [{np.rad2deg(estimated_pose.rotation().ypr()[0]):.2f}, {np.rad2deg(estimated_pose.rotation().ypr()[1]):.2f}, {np.rad2deg(estimated_pose.rotation().ypr()[2]):.2f}]')
        print('--------------------------------')
    elif ground_truth_pose is not None:
        print('--------------------------------')
        print(f'Ground truth translation: ({np.linalg.norm(ground_truth_pose.translation()):.2f} m) [{ground_truth_pose.translation()[0]:.2f}, {ground_truth_pose.translation()[1]:.2f}, {ground_truth_pose.translation()[2]:.2f}]')
        print(f'Ground truth rotation: ({np.linalg.norm(np.rad2deg(ground_truth_pose.rotation().ypr())):.2f} deg) [{np.rad2deg(ground_truth_pose.rotation().ypr()[0]):.2f}, {np.rad2deg(ground_truth_pose.rotation().ypr()[1]):.2f}, {np.rad2deg(ground_truth_pose.rotation().ypr()[2]):.2f}]')
        print('--------------------------------')
    else:
        raise ValueError('At least one of estimated_pose or ground_truth_pose must be provided')


# =========================
# =======  EuRoC  =========
# =========================
import csv
import re
import yaml

# Folder layout expected:
# data/euroc/<SEQ>/mav0/{cam0,cam1,imu0,state_groundtruth_estimate0}
euroc_data_root = Path(__file__).parent / "data" / "euroc"
_euroc_sequence_cache: dict[str, "EuRoCSequence"] = {}

EUROC_WORLD_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)  # ENU, z-up


EUROC_IMU_TO_CV = np.array([
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])


@dataclass
class EuRoCSequence:
    left_paths: list[Path]
    right_paths: list[Path]
    poses_B_from_world: list[gtsam.Pose3]  # T_world_leftCam (physical camera frame)
    frame_ids: list[int]
    calibration: StereoCalibration
    frame_timestamps: np.ndarray  # seconds, aligned L/R
    imu_measurements: list[FrameImuMeasurements] | None
    distortion_model_left: str
    distortion_model_right: str


# ----------- YAML parsing -----------
def se3_inverse(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    
    R_inv = R.T
    t_inv = -R_inv @ t

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv

def _as_4x4(M) -> np.ndarray:
    if isinstance(M, dict) and "data" in M:
        arr = np.asarray(M["data"], dtype=np.float64)
        if arr.size != 16:
            raise ValueError("T_cam_imu data must have 16 elements")
        return arr.reshape(4, 4)
    if isinstance(M, list):
        arr = np.asarray(M, dtype=np.float64)
        if arr.shape == (4, 4):
            return arr
        if arr.size == 16:
            return arr.reshape(4, 4)
    if isinstance(M, str):
        nums = [float(x) for x in re.findall(r"[-+]?[\d.]+(?:e[-+]?\d+)?", M)]
        if len(nums) == 16:
            return np.asarray(nums, dtype=np.float64).reshape(4, 4)
    raise ValueError("Unrecognized 4x4 format for T_cam_imu/T_BS")

def _load_euroc_camera_yaml(yaml_path: Path) -> dict:
    txt = yaml_path.read_text()
    data = yaml.safe_load(txt)

    def _extract(root: dict) -> dict:
        # distortion model
        dmodel = root["distortion_model"]
        assert dmodel == "radial-tangential", "Only radial-tangential distortion model is supported"

        # intrinsics
        fx, fy, cx, cy = [float(x) for x in root["intrinsics"][:4]]
        K = np.array([[fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0]], dtype=np.float64)

        # distortion
        D = np.array(root["distortion_coefficients"])

        # resolution
        w, h = int(root["resolution"][0]), int(root["resolution"][1])

        # extrinsics IMU->cam
        T_imu_from_cam = _as_4x4(root["T_BS"])

        return {
            "K": K, "D": D, "width": w, "height": h,
            "T_imu_from_cam": T_imu_from_cam, "distortion_model": dmodel
        }

    return _extract(data)

# ----------- CSV readers -----------

def _read_euroc_image_index(cam_dir: Path) -> list[tuple[int, Path]]:
    csv_path = cam_dir / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing camera csv at {csv_path}")
    rows: list[tuple[int, Path]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader((line for line in f if not line.startswith("#")))
        for row in reader:
            if len(row) < 2:
                continue
            ts_ns = int(row[0])
            filename = row[1]
            p = cam_dir / "data" / filename
            if p.exists():
                rows.append((ts_ns, p))
    rows.sort(key=lambda x: x[0])
    return rows

def _intersect_pair_index(left_idx: list[tuple[int, Path]], right_idx: list[tuple[int, Path]]) -> tuple[list[Path], list[Path], np.ndarray]:
    L = {ts: p for ts, p in left_idx}
    R = {ts: p for ts, p in right_idx}
    common = sorted(set(L.keys()) & set(R.keys()))
    if not common:
        raise RuntimeError("No common timestamps between cam0 and cam1.")
    left_paths = [L[ts] for ts in common]
    right_paths = [R[ts] for ts in common]
    t_sec = np.asarray([ts * 1e-9 for ts in common], dtype=np.float64)
    return left_paths, right_paths, t_sec

def _read_euroc_imu(imu_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csv_path = imu_dir / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing IMU csv at {csv_path}")
    t_ns, wx, wy, wz, ax, ay, az = [], [], [], [], [], [], []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader((line for line in f if not line.startswith("#")))
        for row in reader:
            if len(row) < 7:
                continue
            t_ns.append(int(row[0]))
            wx.append(float(row[1])); wy.append(float(row[2])); wz.append(float(row[3]))
            ax.append(float(row[4])); ay.append(float(row[5])); az.append(float(row[6]))
    t = np.asarray(t_ns, dtype=np.float64) * 1e-9
    omega = np.column_stack([wx, wy, wz]).astype(np.float64, copy=False)  # rad/s (IMU frame)
    acc = np.column_stack([ax, ay, az]).astype(np.float64, copy=False)    # m/s^2 (IMU frame)
    return t, omega, acc

# GT should all be in the body frame
def _read_euroc_groundtruth(gt_dir: Path) -> tuple[np.ndarray, list[gtsam.Pose3], np.ndarray]:
    csv_path = gt_dir / "data.csv"
    if not csv_path.exists():
        return np.zeros(0, dtype=np.float64), [], np.zeros((0, 3), dtype=np.float64)

    t_ns, px, py, pz, qw, qx, qy, qz, vx, vy, vz = [], [], [], [], [], [], [], [], [], [], []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader((line for line in f if not line.startswith("#")))
        for row in reader:
            if len(row) < 11:
                continue
            t_ns.append(int(row[0]))
            px.append(float(row[1])); py.append(float(row[2])); pz.append(float(row[3]))
            qw.append(float(row[4])); qx.append(float(row[5])); qy.append(float(row[6])); qz.append(float(row[7]))
            vx.append(float(row[8])); vy.append(float(row[9])); vz.append(float(row[10]))
    t = np.asarray(t_ns, dtype=np.float64) * 1e-9
    v = np.column_stack([vx, vy, vz]).astype(np.float64, copy=False)

    poses: list[gtsam.Pose3] = []
    for i in range(len(t)):
        R = gtsam.Rot3.Quaternion(qw[i], qx[i], qy[i], qz[i])
        p = gtsam.Point3(px[i], py[i], pz[i])
        poses.append(gtsam.Pose3(R, p))  # T_imu_from_world
    return t, poses, v


# ----------- Helpers -----------
@dataclass
class EuRoCStereoCalibration(StereoCalibration):
    T_B_from_S0: gtsam.Pose3
    T_B_from_S1: gtsam.Pose3

# how near is nearest?
def _nearest_indices(src_times: np.ndarray, ref_times: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(src_times, ref_times)
    idx = np.clip(idx, 1, len(src_times) - 1)
    left = idx - 1
    right = idx
    pick_left = (ref_times - src_times[left]) <= (src_times[right] - ref_times)
    return np.where(pick_left, left, right)

def _identity_maps(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    return np.meshgrid(xs, ys)

def _stereo_rectify_from_yaml(
    cam0_yaml: Path,
    cam1_yaml: Path,
    alpha: float = 0.0,
) -> tuple[EuRoCStereoCalibration, np.ndarray, np.ndarray, str, str]:
    # Parse both cameras
    c0 = _load_euroc_camera_yaml(cam0_yaml)
    c1 = _load_euroc_camera_yaml(cam1_yaml)
    K0, D0, w0, h0, T_B_from_S0, dm0 = c0["K"], c0["D"], c0["width"], c0["height"], c0["T_imu_from_cam"], c0["distortion_model"]
    K1, D1, w1, h1, T_B_from_S1, dm1 = c1["K"], c1["D"], c1["width"], c1["height"], c1["T_imu_from_cam"], c1["distortion_model"]

    if (w0 != w1) or (h0 != h1):
        raise ValueError("EuRoC left/right image sizes differ; unify before rectification.")

    # Compute stereo extrinsic camera right (1) <- left (0)
    T_S1_from_S0 = se3_inverse(T_B_from_S1) @ T_B_from_S0 
    R_S1_from_S0 = T_S1_from_S0[:3, :3].astype(np.float64, copy=False)
    t_S1_from_S0 = T_S1_from_S0[:3, 3].astype(np.float64, copy=False).reshape(3, 1)
    size = (w0, h0)

    # OpenCV pinhole + radtan
    D0_cv = D0.ravel()[:5].astype(np.float64)
    D1_cv = D1.ravel()[:5].astype(np.float64)
    R0, R1, P0, P1, Q, _, _ = cv2.stereoRectify(
        K0, D0_cv, K1, D1_cv, size, R_S1_from_S0, t_S1_from_S0,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha
    )
    map0x, map0y = cv2.initUndistortRectifyMap(K0, D0_cv, R0, P0, size, cv2.CV_32FC1)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1_cv, R1, P1, size, cv2.CV_32FC1)

    # Build StereoCalibration using rectified intrinsics (from P matrices)
    K0_rect = P0[:3, :3].copy()
    K1_rect = P1[:3, :3].copy()

    calib = EuRoCStereoCalibration(
        K_left=K0.copy(),
        K_right=K1.copy(),
        K_left_rect=K0_rect,
        K_right_rect=K1_rect,
        D_left=D0.copy(),
        D_right=D1.copy(),
        R=R_S1_from_S0.copy(),
        T=t_S1_from_S0.copy(),
        R_left=R0.copy(),
        R_right=R1.copy(),
        P_left=P0.copy(),
        P_right=P1.copy(),
        Q=Q.copy(),
        map_left_x=map0x,
        map_left_y=map0y,
        map_right_x=map1x,
        map_right_y=map1y,
        width=w0,
        height=h0,
        T_B_from_S0=gtsam.Pose3(gtsam.Rot3(T_B_from_S0[:3, :3]), gtsam.Point3(T_B_from_S0[:3, 3])),
        T_B_from_S1=gtsam.Pose3(gtsam.Rot3(T_B_from_S1[:3, :3]), gtsam.Point3(T_B_from_S1[:3, 3])),
    )
    # return both B<-S matrices for later use and distortion models
    return calib, T_B_from_S0, T_B_from_S1, dm0, dm1


# ----------- IMU chunking -----------

def _load_euroc_imu_measurements_camframe(
    imu_t: np.ndarray,
    imu_omega_imu: np.ndarray,           # (N,3) IMU frame
    imu_acc_imu: np.ndarray,             # (N,3) IMU frame (specific force)
    cam_times: np.ndarray,               # (M,) sec
    gt_vel_times: np.ndarray | None = None,   # world frame times
    gt_vel_world: np.ndarray | None = None,   # (K,3) world frame velocities
) -> list[FrameImuMeasurements]:
    """
    Output IMU chunks in LEFT-CAMERA (OpenCV) axes with dts aligned 1:1 to samples.
    We follow the same convention as your TartanAir loader: each IMU sample k carries
    the dt since the previous IMU sample.
    """
    # Global precomputation with a one-sample shift so dt aligns to the *current* sample.
    # After this, indices align: imu_t_s[i] has duration dts_all[i] since the previous raw sample.
    dts_all = np.diff(imu_t)                 # (N-1,)
    imu_t_s = imu_t[1:]                      # (N-1,)
    omega_s = imu_omega_imu[1:, :]           # (N-1,3)
    acc_s   = imu_acc_imu[1:, :]             # (N-1,3)

    out: list[FrameImuMeasurements] = []

    for k in range(len(cam_times)):
        if k == 0:
            out.append(FrameImuMeasurements(
                frame_timestamp=float(cam_times[k]),
                timestamps=np.array([], dtype=np.float64),
                dts=np.array([], dtype=np.float64),
                linear_accelerations=np.array([], dtype=np.float64),
                angular_velocities=np.array([], dtype=np.float64),
            ))
            continue

        t0, t1 = cam_times[k - 1], cam_times[k]

        # Use (t0, t1] semantics to mirror your TartanAir behavior.
        mask = (imu_t_s > t0) & (imu_t_s <= t1)

        ts   = imu_t_s[mask]                 # (n,)
        dts  = dts_all[mask]                 # (n,)  <-- aligned 1:1 with ts
        ome = omega_s[mask]                 # (n,3)
        acc = acc_s[mask]                   # (n,3)

        # TODO: determine if accounting for the translational displacement of the IMU from the camera is necessary

        # Velocity at frame time (world)
        ii = int(_nearest_indices(gt_vel_times, np.array([t1]))[0])
        v_world = gt_vel_world[ii]

        out.append(FrameImuMeasurements(
            frame_timestamp=float(t1),
            timestamps=ts,
            dts=dts,                                # length matches timestamps
            linear_accelerations=acc,
            angular_velocities=ome,
            world_velocity=v_world,
            body_velocity=v_world,
        ))
    return out


# ----------- Sequence load (rectified + conversions) -----------
def _load_euroc_sequence(
    seq_name: str = "MH_01_easy",
    alpha: float = 0.0,
) -> EuRoCSequence:
    if seq_name in _euroc_sequence_cache:
        return _euroc_sequence_cache[seq_name]

    root = euroc_data_root / seq_name / "mav0"
    cam0_dir = root / "cam0"
    cam1_dir = root / "cam1"
    imu_dir = root / "imu0"
    gt_dir = root / "state_groundtruth_estimate0"

    if not cam0_dir.exists() or not cam1_dir.exists():
        raise FileNotFoundError(f"Missing cam0/cam1 at {root}")
    if not (cam0_dir / "sensor.yaml").exists() or not (cam1_dir / "sensor.yaml").exists():
        raise FileNotFoundError("Missing sensor.yaml for cam0/cam1 (required for calibration).")

    # Stereo calibration + extrinsics from EuRoC yaml
    # NOTE: T_B_from_S0, T_B_from_S1 here are actually EuRoC T_BS (IMU <- cam)
    calibration, T_B_from_S0, T_B_from_S1, dml, dmr = _stereo_rectify_from_yaml(
        cam0_dir / "sensor.yaml", cam1_dir / "sensor.yaml", alpha=alpha
    )

    # Intersect timestamps for stereo frames
    left_idx = _read_euroc_image_index(cam0_dir)
    right_idx = _read_euroc_image_index(cam1_dir)
    left_paths, right_paths, cam_times = _intersect_pair_index(left_idx, right_idx)

    # Ground truth poses (world -> IMU) and velocities (world frame)
    gt_t, gt_world_T_imu, gt_vel_world = _read_euroc_groundtruth(gt_dir)

    # EuRoC yaml extrinsic: IMU <- cam0 (sensor to body)
    R_B_from_S0 = T_B_from_S0[:3, :3].astype(np.float64, copy=False)
    t_B_from_S0 = T_B_from_S0[:3, 3].astype(np.float64, copy=False)
    B_from_S0 = gtsam.Pose3(
        gtsam.Rot3(R_B_from_S0),
        gtsam.Point3(t_B_from_S0),
    )

    poses_B_from_world: list[gtsam.Pose3] = []

    # Align each camera timestamp with the closest GT state
    nearest = _nearest_indices(gt_t, cam_times)
    for ii in nearest:
        T_imu_from_world = gt_world_T_imu[ii]  # IMU <- world

        poses_B_from_world.append(T_imu_from_world)

    # IMU stream, in the body frame
    imu_t, imu_omega_imu, imu_acc_imu = _read_euroc_imu(imu_dir)

    # IMU measurements still in the body frame
    imu_meas = _load_euroc_imu_measurements_camframe(
        imu_t,
        imu_omega_imu,
        imu_acc_imu,
        cam_times,
        gt_vel_times=gt_t if gt_t.size > 0 else None,
        gt_vel_world=gt_vel_world if gt_vel_world.size > 0 else None,
    )

    frame_ids = [_parse_frame_id(p, i) for i, p in enumerate(left_paths)]

    seq = EuRoCSequence(
        left_paths=left_paths,
        right_paths=right_paths,
        poses_B_from_world=poses_B_from_world,
        frame_ids=frame_ids,
        calibration=calibration,
        frame_timestamps=cam_times,
        imu_measurements=imu_meas,
        distortion_model_left=dml,
        distortion_model_right=dmr,
    )
    _euroc_sequence_cache[seq_name] = seq
    return seq


def _rectify_and_load(path: Path, mapx: np.ndarray, mapy: np.ndarray) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rect = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rect


# ----------- Public API -----------

def _scale_calibration_for_resize(calib: StereoCalibration, new_w: int, new_h: int) -> StereoCalibration:
    """Return a sterile copy of StereoCalibration with rectified intrinsics scaled to (new_w,new_h).
       Maps are identity since images are already rectified & then resized."""
    old_w, old_h = calib.width, calib.height
    sx, sy = new_w / old_w, new_h / old_h
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)

    K0r = (S @ calib.K_left_rect).copy()
    K1r = (S @ calib.K_right_rect).copy()
    P0 = calib.P_left.copy();  P0[:3, :3] = K0r;  P0[0, 3] *= sx;  P0[1, 3] *= sy
    P1 = calib.P_right.copy(); P1[:3, :3] = K1r;  P1[0, 3] *= sx;  P1[1, 3] *= sy

    map_x, map_y = _identity_maps(new_w, new_h)
    return StereoCalibration(
        K_left=calib.K_left.copy(),
        K_right=calib.K_right.copy(),
        K_left_rect=K0r,
        K_right_rect=K1r,
        D_left=calib.D_left.copy(),
        D_right=calib.D_right.copy(),
        R=calib.R.copy(),
        T=calib.T.copy(),
        R_left=calib.R_left.copy(),
        R_right=calib.R_right.copy(),
        P_left=P0,
        P_right=P1,
        Q=calib.Q.copy(),
        map_left_x=map_x,
        map_left_y=map_y,
        map_right_x=map_x.copy(),
        map_right_y=map_y.copy(),
        width=new_w,
        height=new_h,
    )

# are we rectifying correctly?
def load_euroc_sequence_segment(
    seq_name: str = "MH_01_easy",
    sequence_length: int = 4,
    seed: int = 0,
    sampling_mode: str = "contiguous",   # "contiguous" | "stride"
    min_stride: int = 1,
    max_stride: int | None = None,
    add_imu_noise: bool = False,
    imu_noise_seed: int = 0,
    imu_noise_tau_bias: float = 500.0,
    alpha: float = 0.0,                  # stereoRectify alpha (0=crop more, 1=keep FOV)
    resize_to: tuple[int, int] | None = None,  # e.g., (640, 480) or (640, 640)
) -> FrameSequenceWithGroundTruth[StereoFrame]:
    """Analogous to load_tartanair_sequence_segment, but for EuRoC with proper rectification and IMU/camera conversions."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    if min_stride < 1:
        raise ValueError("min_stride must be >= 1.")
    if max_stride is None:
        max_stride = min_stride
    if max_stride < min_stride:
        raise ValueError("max_stride must be >= min_stride.")

    seq = _load_euroc_sequence(seq_name, alpha=alpha)
    N = len(seq.left_paths)
    if N < sequence_length:
        raise ValueError(f"Sequence too short ({N}) for requested length {sequence_length}.")

    # rng = random.Random(seed)
    # sampling_mode = sampling_mode.lower()
    # if sampling_mode == "contiguous":
    #     max_start = N - sequence_length
    #     start_idx = rng.randint(0, max_start)
    #     sampled = list(range(start_idx, start_idx + sequence_length))
    # elif sampling_mode == "stride":
    #     sampled = []
    #     max_attempts = min(1024, N * 8)
    #     for _ in range(max_attempts):
    #         s = rng.randint(0, N - 1)
    #         cur = s
    #         cand = [cur]
    #         for _ in range(sequence_length - 1):
    #             stride = rng.randint(min_stride, max_stride)
    #             cur += stride
    #             if cur >= N:
    #                 break
    #             cand.append(cur)
    #         if len(cand) == sequence_length:
    #             sampled = cand
    #             break
    #     if len(sampled) != sequence_length:
    #         raise RuntimeError("Could not sample a sequence satisfying stride bounds.")
    # else:
    #     raise ValueError("sampling_mode must be 'contiguous' or 'stride'.")
    sampled = list(range(sequence_length))

    # Optionally scale calibration if resizing
    if resize_to is not None:
        scaled_calib = _scale_calibration_for_resize(seq.calibration, resize_to[0], resize_to[1])
    else:
        scaled_calib = seq.calibration

    # Load rectified frames (then optionally resize)
    frames: list[StereoFrame] = []
    for idx in sampled:
        left = _rectify_and_load(seq.left_paths[idx], seq.calibration.map_left_x, seq.calibration.map_left_y)
        right = _rectify_and_load(seq.right_paths[idx], seq.calibration.map_right_x, seq.calibration.map_right_y)
        if resize_to is not None:
            left = cv2.resize(left, resize_to, interpolation=cv2.INTER_LINEAR)
            right = cv2.resize(right, resize_to, interpolation=cv2.INTER_LINEAR)
        frames.append(RectifiedStereoFrame(left=left, right=right, left_rect=left, right_rect=right, calibration=scaled_calib))

    poses = [seq.poses_B_from_world[i] for i in sampled]
    frame_ids = [seq.frame_ids[i] for i in sampled]

    imu_measurements = (
        [seq.imu_measurements[i] for i in sampled] if seq.imu_measurements is not None else None
    )
    if imu_measurements is not None and add_imu_noise:
        imu_measurements = _apply_sch16t_noise_to_measurements(
            imu_measurements, seed=imu_noise_seed, tau_bias=imu_noise_tau_bias
        )

    frame_timestamps = seq.frame_timestamps[sampled]

    return FrameSequenceWithGroundTruth[StereoFrame](
        frames=frames,
        world_poses=poses,
        frame_indices=sampled,
        frame_ids=frame_ids,
        imu_measurements=imu_measurements,
        full_imu_measurements=seq.imu_measurements,
        frame_timestamps=frame_timestamps,
        ground_truth_depths=None,  # EuRoC has no per-pixel GT depth
    )


def get_euroc_iterator_with_odometry(
    seq_name: str = "MH_01_easy",
    rotation_noise_sigmas: np.ndarray | None = None,
    translation_noise_sigmas: np.ndarray | None = None,
    include_ground_truth: bool = False,
    alpha: float = 0.0,
    resize_to: tuple[int, int] | None = None,
):
    """Yields (StereoFrame, noisy_prev_to_curr [, first_to_curr]) on rectified (and optionally resized) frames."""
    seq = _load_euroc_sequence(seq_name, alpha=alpha)

    if rotation_noise_sigmas is None:
        rotation_noise_sigmas = np.zeros(3, dtype=np.float64)
    else:
        rotation_noise_sigmas = np.asarray(rotation_noise_sigmas, dtype=np.float64).reshape(3)

    if translation_noise_sigmas is None:
        translation_noise_sigmas = np.zeros(3, dtype=np.float64)
    else:
        translation_noise_sigmas = np.asarray(translation_noise_sigmas, dtype=np.float64).reshape(3)

    if resize_to is not None:
        scaled_calib = _scale_calibration_for_resize(seq.calibration, resize_to[0], resize_to[1])
    else:
        scaled_calib = seq.calibration

    world_to_prev = None
    world_to_first = None

    for idx in range(len(seq.left_paths)):
        world_to_cam = seq.poses_B_from_world[idx]

        if world_to_prev is None or world_to_first is None:
            world_to_prev = world_to_cam
            world_to_first = world_to_cam
            continue

        prev_to_curr = world_to_prev.inverse() * world_to_cam

        rot_noise = gtsam.Rot3.Expmap(np.random.normal(0.0, rotation_noise_sigmas))
        trans_noise = gtsam.Point3(np.random.normal(0.0, translation_noise_sigmas))
        noisy_rel = prev_to_curr.compose(gtsam.Pose3(rot_noise, trans_noise))

        first_to_curr = world_to_first.inverse() * world_to_cam
        world_to_prev = world_to_cam

        left = _rectify_and_load(seq.left_paths[idx], seq.calibration.map_left_x, seq.calibration.map_left_y)
        right = _rectify_and_load(seq.right_paths[idx], seq.calibration.map_right_x, seq.calibration.map_right_y)
        if resize_to is not None:
            left = cv2.resize(left, resize_to, interpolation=cv2.INTER_LINEAR)
            right = cv2.resize(right, resize_to, interpolation=cv2.INTER_LINEAR)

        frame = StereoFrame(left=left, right=right, calibration=scaled_calib)

        if include_ground_truth:
            yield frame, noisy_rel, first_to_curr
        else:
            yield frame, noisy_rel


def get_euroc_calibration(seq_name: str = "MH_01_easy") -> StereoCalibration:
    return _load_euroc_sequence(seq_name).calibration


def estimate_world_gravity_from_first_batch(sequence, g=9.81, max_batches=50, normalize_to_g=False):
    """Estimate gravity in the WORLD frame by averaging over multiple IMU batches.

    Args:
        sequence: dataset sequence with .imu_measurements and .world_poses
        g: gravity magnitude
        max_batches: max number of non-empty IMU batches to use
    """
    # return np.array([0.0, 0.0, -g])

    g_world_estimates = []
    weights = []

    # Loop over IMU batches and corresponding world poses
    for i, batch in enumerate(sequence.imu_measurements):
        # Stop if we run out of poses
        if i >= len(sequence.world_poses):
            break

        # Skip empty batches
        if batch.timestamps.size == 0:
            continue

        # Optional cap on how many batches we use
        if len(g_world_estimates) >= max_batches:
            break

        # Mean measured specific force in the (camera) frame
        # (your loader currently rotates IMU -> left-camera/OpenCV)
        a_meas_cam = batch.linear_accelerations.mean(axis=0)  # (3,)

        # world->body at frame i (your world uses the camera’s rotation)
        R_body_from_world = sequence.world_poses[i].rotation().matrix()  # maps world -> body

        g_world_i = -R_body_from_world @ a_meas_cam

        # Skip degenerate estimates
        norm_i = np.linalg.norm(g_world_i)
        if norm_i < 1e-6:
            continue

        g_world_estimates.append(g_world_i)
        # Weight by number of IMU samples in the batch
        weights.append(float(batch.timestamps.size))

    if not g_world_estimates:
        raise ValueError("No valid IMU batches found to estimate gravity.")

    # Weighted average of all per-frame estimates in world frame
    g_world_avg = np.average(np.stack(g_world_estimates, axis=0), axis=0, weights=weights)

    # Normalize to have magnitude |g|
    if normalize_to_g:
        g_world_avg = (g_world_avg / np.linalg.norm(g_world_avg)) * float(g)

    # compute standard deviation of the estimates
    g_world_std = np.std(np.stack(g_world_estimates, axis=0), axis=0)

    print(f"Gravity estimate: {g_world_avg}, standard deviation: {g_world_std}")
    print(f"Norm: {np.linalg.norm(g_world_avg)}")

    return g_world_avg
