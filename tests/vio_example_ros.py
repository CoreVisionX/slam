import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.image import message_to_cvimage

from slam.vio import save_tum_sequence
from slam.vio.core import VIO


SHIFT_CAM0_TO_IMU = 0.0


def load_rosbag_sequence(
    bag_path: Path,
    left_topic: str,
    right_topic: str,
    imu_topic: str,
    max_time_diff: float = 0.01,
    source_fps: int = 60,
    target_fps: int = 20,
):
    """
    Load distorted stereo + IMU from ROS1 bag using rosbags/rosbags-image.

    Args:
        bag_path: Path to rosbag.
        left_topic: Left camera image topic.
        right_topic: Right camera image topic.
        imu_topic: IMU topic.
        max_time_diff: Maximum allowed |t_left - t_right| in seconds for a
                       stereo pair to be considered valid.

    Returns:
        frames: list of (t_cam, left_img, right_img)  [distorted, RGB]
        imu_ts: (N,) float64  [aligned to camera time via Kalibr shift]
        imu_acc: (N, 3) float64, m/s^2
        imu_gyro: (N, 3) float64, rad/s
    """
    bag_path = Path(bag_path)
    if not bag_path.exists():
        raise FileNotFoundError(bag_path)

    typestore = get_typestore(Stores.ROS1_NOETIC)

    left_msgs = []
    right_msgs = []
    imu_msgs = []

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        left_conns = [c for c in reader.connections if c.topic == left_topic]
        right_conns = [c for c in reader.connections if c.topic == right_topic]
        imu_conns = [c for c in reader.connections if c.topic == imu_topic]

        if not left_conns:
            raise RuntimeError(f"No connections found for left_topic={left_topic}")
        if not right_conns:
            raise RuntimeError(f"No connections found for right_topic={right_topic}")
        if not imu_conns:
            raise RuntimeError(f"No connections found for imu_topic={imu_topic}")

        conns = left_conns + right_conns + imu_conns

        for connection, timestamp, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)

            if connection.topic == left_topic:
                st = msg.header.stamp
                t = float(st.sec) + float(st.nanosec) * 1e-9
                img = message_to_cvimage(msg)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                left_msgs.append((t, img))

            elif connection.topic == right_topic:
                st = msg.header.stamp
                t = float(st.sec) + float(st.nanosec) * 1e-9
                img = message_to_cvimage(msg)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                right_msgs.append((t, img))

            elif connection.topic == imu_topic:
                st = msg.header.stamp
                t = float(st.sec) + float(st.nanosec) * 1e-9
                imu_msgs.append((t, msg))

    if not left_msgs or not right_msgs:
        raise RuntimeError("No stereo frames found in bag.")

    # Sort and pair stereo frames with timestamp margin
    left_msgs.sort(key=lambda x: x[0])
    right_msgs.sort(key=lambda x: x[0])

    frames = []
    i = j = 0
    while i < len(left_msgs) and j < len(right_msgs):
        t_l, img_l = left_msgs[i]
        t_r, img_r = right_msgs[j]
        dt = t_l - t_r

        if abs(dt) <= max_time_diff:
            # Accept as a stereo pair
            t = 0.5 * (t_l + t_r)
            frames.append((t, img_l, img_r))
            i += 1
            j += 1
        elif dt < 0:
            # Left frame is earlier; try next left
            i += 1
        else:
            # Right frame is earlier; try next right
            j += 1

    if not frames:
        raise RuntimeError(
            f"No stereo pairs found with |Δt| <= {max_time_diff} s "
            f"(left={len(left_msgs)}, right={len(right_msgs)})"
        )

    # downsample frames to target FPS
    ratio = source_fps // target_fps
    frames = frames[::ratio]

    # Process IMU messages into arrays
    imu_msgs.sort(key=lambda x: x[0])

    imu_ts_list = []
    acc_list = []
    gyro_list = []

    for t, msg in imu_msgs:
        la = msg.linear_acceleration
        av = msg.angular_velocity
        imu_ts_list.append(t)
        acc_list.append([la.x, la.y, la.z])
        gyro_list.append([av.x, av.y, av.z])

    imu_ts = np.asarray(imu_ts_list, dtype=np.float64)
    imu_acc = np.asarray(acc_list, dtype=np.float64)
    imu_gyro = np.asarray(gyro_list, dtype=np.float64)

    # Apply Kalibr time offset: t_imu_aligned = t_imu_raw + shift_cam0_to_imu
    imu_ts = imu_ts + SHIFT_CAM0_TO_IMU

    print(
        f"Loaded from {bag_path}:\n"
        f"  stereo frames (paired with |Δt| <= {max_time_diff}s): {len(frames)}\n"
        f"  imu samples:   {imu_ts.shape[0]}"
    )

    return frames, imu_ts, imu_acc, imu_gyro


def run_vio_rosbag_kalibr(
    vio_config_path: Path,
    bag_path: Path,
    left_topic: str,
    right_topic: str,
    imu_topic: str,
    output_path: Path,
) -> None:
    # Load bag data
    frames, imu_ts, imu_acc, imu_gyro = load_rosbag_sequence(
        bag_path=bag_path,
        left_topic=left_topic,
        right_topic=right_topic,
        imu_topic=imu_topic,
    )

    # skip the first frame
    frames = frames[1:]

    if len(frames) < 2:
        raise RuntimeError("Not enough frames for VIO (need at least 2).")

    # 3) Instantiate VIO from base config and override K/baseline
    vio: VIO = VIO.from_config(vio_config_path)

    # 4) Initialize VIO with first rectified stereo frame
    t0, left0_rect, right0_rect = frames[0]

    vio.reset(
        timestamp=float(t0),
        left_rect=left0_rect,
        right_rect=right0_rect,
        t=np.zeros(3, dtype=np.float64),
        R=np.eye(3, dtype=np.float64),
        v=np.zeros(3, dtype=np.float64),
    )

    # 5) Main VIO loop
    num_frames = len(frames)
    imu_idx = 0
    num_imu = imu_ts.shape[0]

    print(f"Running VIO on bag: {num_frames} frames, {num_imu} IMU samples")

    pbar = tqdm(range(1, num_frames), desc="Estimating trajectory (0.0 m)")
    for k in pbar:
        t_prev = frames[k - 1][0]
        t_curr, left_rect, right_rect = frames[k]

        # Collect IMU samples in (t_prev, t_curr]
        acc_batch: List[np.ndarray] = []
        gyro_batch: List[np.ndarray] = []
        ts_batch: List[float] = []

        while imu_idx < num_imu and imu_ts[imu_idx] <= t_curr:
            t = float(imu_ts[imu_idx])
            if t > t_prev:
                ts_batch.append(t)
                acc_batch.append(imu_acc[imu_idx])
                gyro_batch.append(imu_gyro[imu_idx])
            imu_idx += 1
        
        if len(ts_batch) < 2:
            print("not enough imu samples, skipping frame")
            continue
        else:
            batch_ts = np.asarray(ts_batch, dtype=np.float64)
            imu_ts_rel = batch_ts  # keep absolute; your ImuPreintegrator uses dt differences anyway

            batch_acc = np.asarray(acc_batch, dtype=np.float64)
            batch_gyro = np.asarray(gyro_batch, dtype=np.float64)

            vio.process(
                timestamp=float(t_curr),
                left_rect=left_rect,
                right_rect=right_rect,
                imu_acc=batch_acc,
                imu_gyro=batch_gyro,
                imu_ts=imu_ts_rel,
            )

        pbar.set_description(f"Estimating trajectory ({vio.get_distance_traveled():.2f} m)")

    print("\nFinished VIO on bag.")

    traj = vio.get_estimated_trajectory()
    save_tum_sequence(traj, str(output_path))
    print(f"Saved trajectory to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag",
        type=Path,
        default="data/living27.bag",
        help="Path to ROS1 bag file.",
    )
    parser.add_argument(
        "--vio-config",
        type=Path,
        default="config/vio_d435i.yaml",
        help="Path to base VIO config YAML (with Kalibr extrinsics & IMU params).",
    )
    parser.add_argument(
        "--left-topic",
        type=str,
        default="/camera/camera/infra1/image_rect_raw",
        help="Left camera topic (distorted).",
    )
    parser.add_argument(
        "--right-topic",
        type=str,
        default="/camera/camera/infra2/image_rect_raw",
        help="Right camera topic (distorted).",
    )
    parser.add_argument(
        "--imu-topic",
        type=str,
        default="/camera/camera/imu",
        help="IMU topic (sensor_msgs/Imu).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vio_estimated_d435_kalibr.txt"),
        help="Where to write the TUM trajectory.",
    )
    args = parser.parse_args()

    run_vio_rosbag_kalibr(
        vio_config_path=args.vio_config,
        bag_path=args.bag,
        left_topic=args.left_topic,
        right_topic=args.right_topic,
        imu_topic=args.imu_topic,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
