from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, Tuple, Generic, TypeVar

import cv2
import gtsam
from registration.registration import StereoCalibration, StereoFrame, FramePairWithGroundTruth
import tartanair as ta
from util import convert_coordinate_frame, se3_flattened_to_pose3
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
tartanair_calib = StereoCalibration.create(
    K=np.array([
        [WIDTH / 2, 0.0, WIDTH / 2],
        [0.0, HEIGHT / 2, HEIGHT / 2],
        [0.0, 0.0, 1.0]
    ]),
    T=np.array([0.0, 0.25, 0.0]),
    R=np.eye(3),
    width=WIDTH, height=HEIGHT
)

@dataclass
class TartanAirSequence:
    left_paths: list[Path]
    right_paths: list[Path]
    poses: list[gtsam.Pose3]
    frame_ids: list[int]

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

    @property
    def length(self) -> int:
        return len(self.frames)

    def relative_pose(self, first_idx: int, second_idx: int) -> gtsam.Pose3:
        if first_idx < 0 or second_idx < 0:
            raise ValueError("Frame indices must be non-negative.")
        if first_idx >= self.length or second_idx >= self.length:
            raise IndexError("Frame index out of range for this sequence.")
        return self.world_poses[first_idx].inverse() * self.world_poses[second_idx]


def _parse_frame_id(path: Path, fallback: int) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            pass
    return fallback

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

    sequence = TartanAirSequence(
        left_paths=left_paths,
        right_paths=right_paths,
        poses=poses,
        frame_ids=frame_ids,
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



def load_tartanair_sequence_segment(
    env: str = "ArchVizTinyHouseDay",
    difficulty: str = "easy",
    traj: str = "P000",
    sequence_length: int = 4,
    seed: int = 0,
    sampling_mode: str = "contiguous",
    min_stride: int = 1,
    max_stride: int | None = None,
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

    return FrameSequenceWithGroundTruth[StereoFrame](
        frames=frames,
        world_poses=poses,
        frame_indices=sampled_indices,
        frame_ids=frame_ids,
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
