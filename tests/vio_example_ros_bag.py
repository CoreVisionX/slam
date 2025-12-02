import argparse
from collections import deque
from pathlib import Path
import time
from typing import Deque, Iterable

import cv2
import numpy as np
import multiprocessing as mp

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.image import message_to_cvimage

from slam.vio import save_tum_sequence
from slam.vio.mp_runner import AsyncVIO


SHIFT_CAM0_TO_IMU = 0.0


def iter_rosbag_stream(
    bag_path: Path,
    left_topic: str,
    right_topic: str,
    imu_topic: str,
    *,
    max_time_diff: float = 0.01,
    source_fps: int = 20,
    target_fps: int = 20,
    take_every: int = 3,  # take every i-th frame
) -> Iterable[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Stream stereo pairs + IMU batches from a ROS1 bag without preloading everything.
    Yields (t_frame, left_img, right_img, imu_ts, imu_acc, imu_gyro).
    """
    bag_path = Path(bag_path)
    if not bag_path.exists():
        raise FileNotFoundError(bag_path)

    typestore = get_typestore(Stores.ROS1_NOETIC)
    frame_downsample = max(1, source_fps // target_fps)

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

        left_queue: Deque[tuple[float, np.ndarray]] = deque()
        right_queue: Deque[tuple[float, np.ndarray]] = deque()
        imu_queue: Deque[tuple[float, np.ndarray, np.ndarray]] = deque()

        frame_count_left = 0
        frame_count_right = 0
        pair_idx = 0
        last_output_ts: float | None = None

        for connection, _, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)

            if connection.topic == left_topic:
                if frame_count_left % take_every != 0:
                    frame_count_left += 1
                    continue

                st = msg.header.stamp
                t = float(st.sec) + float(st.nanosec) * 1e-9
                img = message_to_cvimage(msg)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                left_queue.append((t, img))
                frame_count_left += 1

            elif connection.topic == right_topic:
                if frame_count_right % take_every != 0:
                    frame_count_right += 1
                    continue

                st = msg.header.stamp
                t = float(st.sec) + float(st.nanosec) * 1e-9
                img = message_to_cvimage(msg)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                right_queue.append((t, img))
                frame_count_right += 1

            elif connection.topic == imu_topic:
                st = msg.header.stamp
                t = float(st.sec) + float(st.nanosec) * 1e-9
                imu_queue.append(
                    (
                        t + SHIFT_CAM0_TO_IMU,
                        np.asarray([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], dtype=np.float64),
                        np.asarray([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=np.float64),
                    )
                )

            # Try to form stereo pairs whenever both queues have data
            while left_queue and right_queue:
                t_l, img_l = left_queue[0]
                t_r, img_r = right_queue[0]
                dt = t_l - t_r

                if abs(dt) <= max_time_diff:
                    left_queue.popleft()
                    right_queue.popleft()
                    t_pair = 0.5 * (t_l + t_r)
                    pair_idx += 1
                    should_output = (pair_idx - 1) % frame_downsample == 0

                    # Drop IMUs older than last output to keep queue bounded
                    if last_output_ts is not None:
                        while imu_queue and imu_queue[0][0] <= last_output_ts:
                            imu_queue.popleft()

                    if should_output:
                        ts_list: list[float] = []
                        acc_list: list[np.ndarray] = []
                        gyro_list: list[np.ndarray] = []
                        while imu_queue and imu_queue[0][0] <= t_pair:
                            t_imu, acc, gyro = imu_queue.popleft()
                            ts_list.append(t_imu)
                            acc_list.append(acc)
                            gyro_list.append(gyro)

                        last_output_ts = t_pair

                        imu_ts = np.asarray(ts_list, dtype=np.float64)
                        imu_acc = np.asarray(acc_list, dtype=np.float64)
                        imu_gyro = np.asarray(gyro_list, dtype=np.float64)

                        yield (t_pair, img_l, img_r, imu_ts, imu_acc, imu_gyro)
                elif dt < 0:
                    left_queue.popleft()
                else:
                    right_queue.popleft()


def run_vio_rosbag_kalibr(
    vio_config_path: Path,
    bag_path: Path,
    left_topic: str,
    right_topic: str,
    imu_topic: str,
    output_path: Path,
) -> None:
    stream = iter_rosbag_stream(
        bag_path=bag_path,
        left_topic=left_topic,
        right_topic=right_topic,
        imu_topic=imu_topic,
    )

    try:
        first = next(stream)
    except StopIteration:
        raise RuntimeError("No stereo pairs found in bag.")

    t0, left0_rect, right0_rect, _, _, _ = first

    async_vio = AsyncVIO(vio_config_path=vio_config_path)
    
    print("Waiting for 5 seconds for the VIO worker to start")
    time.sleep(5)

    async_vio.reset(
        timestamp=float(t0),
        left_rect=left0_rect,
        right_rect=right0_rect,
        t=np.zeros(3, dtype=np.float64),
        R=np.eye(3, dtype=np.float64),
        v=np.zeros(3, dtype=np.float64),
    )

    print("Running VIO on bag (async streaming)...")
    frames_enqueued = 0
    start_time = time.perf_counter()

    for i, (t_curr, left_rect, right_rect, imu_ts, imu_acc, imu_gyro) in enumerate(stream):
        if imu_ts.size < 2:
            print("not enough imu samples, skipping frame")
            continue
        enqueue_start = time.perf_counter()

        # process imu samples sequentially
        for t_imu, acc, gyro in zip(imu_ts, imu_acc, imu_gyro):
            async_vio.process_imu(imu_ts=np.array([t_imu]), imu_acc=np.array([acc]), imu_gyro=np.array([gyro]))

        async_vio.process(
            timestamp=float(t_curr),
            left_rect=left_rect,
            right_rect=right_rect,
        )

        frames_enqueued += 1
        if frames_enqueued % 100 == 0:
            elapsed = time.perf_counter() - start_time
            enqueue_dt = time.perf_counter() - enqueue_start
            print(
                f"Enqueued {frames_enqueued} frames in {elapsed:.2f}s ({frames_enqueued/elapsed:.2f} Hz), "
                f"last enqueue latency {enqueue_dt*1000:.2f} ms"
            )

        # wait to simulate real-time processing
        time.sleep(0.05)

    traj = async_vio.finish()
    elapsed_total = time.perf_counter() - start_time
    print(f"\nFinished VIO on bag. Total time {elapsed_total:.2f}s for {frames_enqueued} frames.")

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
    mp.set_start_method("spawn")

    main()
