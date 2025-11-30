import argparse
import sys
from pathlib import Path
from collections import deque
import time
from threading import Lock

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from slam.vio.core import VIO

def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9

class VioRos2Node(Node):
    def __init__(self, vio_config_path, left_topic, right_topic, imu_topic, output_path):
        super().__init__("vio_node")
        
        self.vio = VIO.from_config(vio_config_path)
        self.output_path = output_path
        self.bridge = CvBridge()
        
        # ---------------- FIX 1: Active Frame Dropping Logic ----------------
        # We need separate locks because we don't want the IMU callback 
        # waiting on the VIO thread to finish a heavy calculation.
        self.imu_lock = Lock()
        self.img_lock = Lock()
        
        self.imu_queue = deque()
        self.image_queue = deque()
        
        self.last_frame_time = None
        self.initialized = False

        # ---------------- FIX 2: Callback Groups ----------------
        self.sensor_group = MutuallyExclusiveCallbackGroup()
        self.vio_group = MutuallyExclusiveCallbackGroup()

        # ---------------- FIX 3: Massive QoS Depth ----------------
        # 200 was too small. At 200Hz, that's only 1 second of buffer.
        # If Python hangs for 1.1 seconds, you lose data.
        # Set to 2000 (10 seconds) to be safe against GC pauses/GIL contention.
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST, 
            depth=2000 
        )
        
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST, 
            depth=10 
        )

        # IMU Sub
        self.create_subscription(Imu, imu_topic, self.imu_callback, qos_profile=imu_qos, callback_group=self.sensor_group)

        # Camera Subs
        self.left_sub = Subscriber(self, Image, left_topic, qos_profile=cam_qos, callback_group=self.sensor_group)
        self.right_sub = Subscriber(self, Image, right_topic, qos_profile=cam_qos, callback_group=self.sensor_group)
        self.sync = ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=20, slop=0.05)
        self.sync.registerCallback(self.stereo_callback)

        # Processing Timer (Runs on VIO group)
        self.create_timer(0.001, self.run_vio_loop, callback_group=self.vio_group)

        self.fps_last_time = self.get_clock().now().nanoseconds * 1e-9
        self.fps_frame_count = 0

        self.get_logger().info("VIO Node initialized")

    # ---------------- LIGHTWEIGHT CALLBACKS ----------------
    # GOAL: Get in and out of Python as fast as possible to release GIL.

    def imu_callback(self, msg: Imu) -> None:
        # ---------------- FIX 4: No Numpy in Callback ----------------
        # Allocating numpy arrays 200x/sec is expensive in Python.
        # Just store the raw object or tuple. Convert later in the worker.
        t = stamp_to_sec(msg.header.stamp)
        with self.imu_lock:
            self.imu_queue.append((t, msg))

    def stereo_callback(self, left_msg: Image, right_msg: Image) -> None:
        t = stamp_to_sec(left_msg.header.stamp)
        with self.img_lock:
            # If queue is getting too big, we are falling behind.
            # Drop the OLDEST frame to catch up to the newest.
            if len(self.image_queue) > 5:
                self.image_queue.popleft() 
                # Optional: Log warning, but keep it quiet to avoid I/O blocking
            
            self.image_queue.append((t, left_msg, right_msg))

    # ---------------- HEAVY PROCESSING LOOP ----------------

    def run_vio_loop(self):
        # 1. Check if we have an image (Snapshot with lock)
        img_data = None
        with self.img_lock:
            if not self.image_queue: return
            img_data = self.image_queue[0] # Peek

        img_t = img_data[0]

        # 2. Check if we have enough IMU data (Snapshot with lock)
        has_future_imu = False
        with self.imu_lock:
            if self.imu_queue:
                if self.imu_queue[-1][0] >= img_t:
                    has_future_imu = True
        
        if not has_future_imu:
            return # Yield, let callbacks run

        # 3. We have data! Extract it.
        process_data = None
        
        # Pop Image
        with self.img_lock:
            if self.image_queue: # Check again
                self.image_queue.popleft()
                process_data = {
                    't': img_t,
                    'left': img_data[1],
                    'right': img_data[2]
                }

        if not process_data: return

        # Pop IMU Batch
        ts_list, acc_list, gyro_list = [], [], []
        
        t_prev = self.last_frame_time if self.last_frame_time is not None else -1.0
        
        with self.imu_lock:
            # Discard old
            while self.imu_queue and self.imu_queue[0][0] <= t_prev:
                self.imu_queue.popleft()
            
            # Collect new
            # NOTE: We convert to numpy HERE, in the slow thread, not the callback
            for t_imu, msg in self.imu_queue:
                if t_imu > img_t: break
                
                ts_list.append(t_imu)
                acc_list.append([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
                gyro_list.append([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        # ---------------- CRITICAL LOGIC: Handle "Holes" ----------------
        if len(ts_list) == 0 and self.initialized:
            self.get_logger().warn("Dropped frame due to missing IMU data (GIL starvation?).")
            return

        # Prepare numpy arrays
        imu_ts = np.array(ts_list, dtype=np.float64)
        imu_acc = np.array(acc_list, dtype=np.float64)
        imu_gyro = np.array(gyro_list, dtype=np.float64)

        # 4. Decode Images (Heavy I/O)
        left = self.bridge.imgmsg_to_cv2(process_data['left'], desired_encoding="rgb8")
        right = self.bridge.imgmsg_to_cv2(process_data['right'], desired_encoding="rgb8")
        
        # 5. Process VIO (Heavy Math)
        start_compute = time.perf_counter()
        
        if not self.initialized:
            self.vio.reset(
                timestamp=process_data['t'],
                left_rect=left, right_rect=right,
                t=np.zeros(3), R=np.eye(3), v=np.zeros(3)
            )
            self.initialized = True
        else:
            self.vio.process(
                timestamp=process_data['t'],
                left_rect=left,
                right_rect=right,
                imu_acc=imu_acc,
                imu_gyro=imu_gyro,
                imu_ts=imu_ts
            )
        
        self.last_frame_time = process_data['t']
        
        # 6. Check if we are too slow
        compute_time = time.perf_counter() - start_compute
        if compute_time > 0.05: # Warn if VIO takes > 50ms
             self.get_logger().warn(f"VIO SLOW: {compute_time:.4f}s", throttle_duration_sec=2.0)

        # Logging
        self.fps_frame_count += 1
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.fps_last_time >= 1.0:
            fps = self.fps_frame_count / (now - self.fps_last_time)
            self.get_logger().info(f"VIO FPS: {fps:.2f}", throttle_duration_sec=2.0)
            self.fps_frame_count = 0
            self.fps_last_time = now

            # Log distance traveled
            dist = self.vio.get_distance_traveled()
            self.get_logger().info(f"Distance traveled: {dist:.2f} m", throttle_duration_sec=5.0)

    # ---------------- Shutdown handling ----------------

    def save_trajectory(self) -> None:
        traj = self.vio.get_estimated_trajectory()
        save_tum_sequence(traj, str(self.output_path))
        self.get_logger().info(f"Saved VIO trajectory to {self.output_path}")


def main(argv=None) -> None:
    # Parse our own CLI args (leaving ROS remapping args alone)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--vio-config",
        type=Path,
        default=Path("config/vio_d435i.yaml"),
        help="Path to VIO config YAML.",
    )
    parser.add_argument(
        "--left-topic",
        type=str,
        default="/camera/camera/infra1/image_rect_raw",
        help="Left rectified image topic.",
    )
    parser.add_argument(
        "--right-topic",
        type=str,
        default="/camera/camera/infra2/image_rect_raw",
        help="Right rectified image topic.",
    )
    parser.add_argument(
        "--imu-topic",
        type=str,
        default="/camera/camera/imu",
        help="IMU topic.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vio_estimated_d435_live.txt"),
        help="Path to save TUM trajectory on shutdown.",
    )

    args, ros_args = parser.parse_known_args(argv)

    rclpy.init(args=ros_args)

    node = VioRos2Node(
        vio_config_path=args.vio_config,
        left_topic=args.left_topic,
        right_topic=args.right_topic,
        imu_topic=args.imu_topic,
        output_path=args.output,
    )

    # Use multithreaded executor so callbacks can interrupt/run parallel
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down node...")
    finally:
        node.save_trajectory()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])