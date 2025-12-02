import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

from slam.vio.mp_runner import AsyncVIO


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class VioRos2Node(Node):
    def __init__(
        self,
        vio_config_path: Path,
        left_topic: str,
        right_topic: str,
        imu_topic: str,
        output_path: Path,
        frame_stride: int = 1,
    ):
        super().__init__("vio_node")
        self.bridge = CvBridge()
        self.output_path = output_path
        self.frame_stride = max(1, frame_stride)
        self.max_time_diff = 0.01

        self.async_vio = AsyncVIO(vio_config_path)
        self.initialized = False
        self.pair_idx = 0

        # QoS and subscriptions
        self.sensor_group = MutuallyExclusiveCallbackGroup()
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=4000,
        )
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        self.create_subscription(Imu, imu_topic, self.imu_callback, qos_profile=imu_qos, callback_group=self.sensor_group)
        self.left_sub = Subscriber(self, Image, left_topic, qos_profile=cam_qos, callback_group=self.sensor_group)
        self.right_sub = Subscriber(self, Image, right_topic, qos_profile=cam_qos, callback_group=self.sensor_group)
        self.sync = ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=50, slop=self.max_time_diff)
        self.sync.registerCallback(self.stereo_callback)

        self.fps_last_time = time.perf_counter()
        self.fps_frame_count = 0

        self.get_logger().info("VIO Node initialized")

    def imu_callback(self, msg: Imu) -> None:
        t = stamp_to_sec(msg.header.stamp)
        acc = np.asarray(
            [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            dtype=np.float64,
        )
        gyro = np.asarray(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            dtype=np.float64,
        )
        # push single-sample batch into IMU ring buffer
        self.async_vio.process_imu(
            imu_ts=np.asarray([t], dtype=np.float64),
            imu_acc=acc.reshape(1, 3),
            imu_gyro=gyro.reshape(1, 3),
        )

    def stereo_callback(self, left_msg: Image, right_msg: Image) -> None:
        t_l = stamp_to_sec(left_msg.header.stamp)
        t_r = stamp_to_sec(right_msg.header.stamp)
        t_pair = 0.5 * (t_l + t_r)

        self.pair_idx += 1
        if (self.pair_idx - 1) % self.frame_stride != 0:
            return

        left_gray = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding="mono8")
        right_gray = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding="mono8")
        left = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2RGB)
        right = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2RGB)
        
        if not self.initialized:
            try:
                self.async_vio.reset(
                    timestamp=float(t_pair),
                    left_rect=left,
                    right_rect=right,
                    t=np.zeros(3, dtype=np.float64),
                    R=np.eye(3, dtype=np.float64),
                    v=np.zeros(3, dtype=np.float64),
                )
                self.initialized = True
            except Exception as exc:
                self.get_logger().error(f"VIO reset failed: {exc}")
                return
        else:
            try:
                self.async_vio.process(
                    timestamp=float(t_pair),
                    left_rect=left,
                    right_rect=right,
                )
            except Exception as exc:
                self.get_logger().error(f"VIO process failed: {exc}")
                return

        self._log_fps()

    def _log_fps(self):
        self.fps_frame_count += 1
        now = time.perf_counter()
        dt = now - self.fps_last_time
        if dt >= 1.0:
            fps = self.fps_frame_count / dt
            self.get_logger().info(f"Enqueued FPS: {fps:.2f}")
            self.fps_frame_count = 0
            self.fps_last_time = now

    def shutdown(self):
        self.async_vio.stop()
        self.get_logger().info("Shutdown complete.")

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vio-config", type=Path, default=Path("config/vio_d435i.yaml"))
    parser.add_argument("--left-topic", type=str, default="/camera/camera/infra1/image_rect_raw")
    parser.add_argument("--right-topic", type=str, default="/camera/camera/infra2/image_rect_raw")
    parser.add_argument("--imu-topic", type=str, default="/camera/camera/imu")
    parser.add_argument("--output", type=Path, default=Path("vio_estimated_d435_live.txt"))
    parser.add_argument("--frame-stride", type=int, default=2)
    args, ros_args = parser.parse_known_args(argv)

    rclpy.init(args=ros_args)

    node = VioRos2Node(
        vio_config_path=args.vio_config,
        left_topic=args.left_topic,
        right_topic=args.right_topic,
        imu_topic=args.imu_topic,
        output_path=args.output,
        frame_stride=args.frame_stride,
    )

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down node...")
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
