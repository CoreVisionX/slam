import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import cv2
from PIL import Image
from term_image import disable_queries
from term_image.image import BaseImage, KittyImage

from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

from slam.vio import rs_sdk

BaseImage.forced_support = True  # force support for all term image formats
KittyImage.set_render_method("WHOLE") # render the whole image at once


def numpy_to_ros_image(img: np.ndarray, timestamp: float, encoding: str = "mono8"):
    """Convert numpy array to ROS sensor_msgs/Image message."""
    if len(img.shape) == 2:
        # Grayscale image
        height, width = img.shape
        is_bigendian = 0
        step = width
        # Ensure data is a numpy array of uint8
        data = np.asarray(img, dtype=np.uint8).flatten()
    elif len(img.shape) == 3:
        # Color image - convert to grayscale for mono8
        if encoding == "mono8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            height, width = img.shape
            is_bigendian = 0
            step = width
            # Ensure data is a numpy array of uint8
            data = np.asarray(img, dtype=np.uint8).flatten()
        else:
            height, width, channels = img.shape
            is_bigendian = 0
            step = width * channels
            # Ensure data is a numpy array of uint8
            data = np.asarray(img, dtype=np.uint8).flatten()
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    
    # Convert timestamp to ROS Time
    sec = int(timestamp)
    nanosec = int((timestamp - sec) * 1e9)
    
    # Create message object using SimpleNamespace (matches rosbags message structure)
    msg = SimpleNamespace()
    msg.header = SimpleNamespace()
    msg.header.seq = 0  # Sequence number (not critical for bag files)
    msg.header.stamp = SimpleNamespace()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    msg.header.frame_id = "camera"
    msg.height = height
    msg.width = width
    msg.encoding = encoding
    msg.is_bigendian = is_bigendian
    msg.step = step
    msg.data = data
    
    return msg


def calibrate_camera(
    images: list[np.ndarray],
    checkerboard_size: tuple[int, int],
    square_size: float,
    disable_distortion: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Calibrate camera using checkerboard pattern.
    
    Returns:
        K: Camera matrix (3x3)
        D: Distortion coefficients (5x1 or zeros if disabled)
        reprojection_error: RMS reprojection error
    """
    # Prepare object points (3D points in real world space)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3d points in real world space
    imgpoints_left = []  # 2d points in image plane (left camera)
    imgpoints_right = []  # 2d points in image plane (right camera)
    
    # Find checkerboard corners in images
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    print(f"\nCalibrating cameras using {len(images)} image pairs...")
    print(f"Looking for checkerboard pattern: {checkerboard_size[0]}x{checkerboard_size[1]} inner corners")
    
    valid_pairs = 0
    for i, (left_img, right_img) in enumerate(images):
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img
        
        # Find corners in both images
        ret_left, corners_left = cv2.findChessboardCorners(
            left_gray, checkerboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        ret_right, corners_right = cv2.findChessboardCorners(
            right_gray, checkerboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret_left and ret_right:
            # Refine corner positions
            corners_left_refined = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
            corners_right_refined = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints_left.append(corners_left_refined)
            imgpoints_right.append(corners_right_refined)
            valid_pairs += 1
            
            if (valid_pairs % 10) == 0:
                print(f"  Found {valid_pairs} valid checkerboard pairs...")
    
    if valid_pairs < 3:
        raise ValueError(f"Not enough valid checkerboard images found ({valid_pairs}). Need at least 3.")
    
    print(f"Using {valid_pairs} valid checkerboard pairs for calibration")
    
    # Calibrate left camera
    img_size = (images[0][0].shape[1], images[0][0].shape[0])
    
    if disable_distortion:
        # Use zero distortion model
        flags = cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_ZERO_TANGENT_DIST
    else:
        flags = 0
    
    ret_left, K_left, D_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, img_size, None, None, flags=flags
    )
    
    # Calibrate right camera
    ret_right, K_right, D_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, img_size, None, None, flags=flags
    )
    
    if not ret_left or not ret_right:
        raise RuntimeError("Camera calibration failed")
    
    # Calculate reprojection error for left camera
    total_error_left = 0
    for i in range(len(objpoints)):
        imgpoints2_left, _ = cv2.projectPoints(objpoints[i], rvecs_left[i], tvecs_left[i], K_left, D_left)
        error = cv2.norm(imgpoints_left[i], imgpoints2_left, cv2.NORM_L2) / len(imgpoints2_left)
        total_error_left += error
    reproj_error_left = total_error_left / len(objpoints)
    
    # Calculate reprojection error for right camera
    total_error_right = 0
    for i in range(len(objpoints)):
        imgpoints2_right, _ = cv2.projectPoints(objpoints[i], rvecs_right[i], tvecs_right[i], K_right, D_right)
        error = cv2.norm(imgpoints_right[i], imgpoints2_right, cv2.NORM_L2) / len(imgpoints2_right)
        total_error_right += error
    reproj_error_right = total_error_right / len(objpoints)
    
    return (K_left, D_left, reproj_error_left), (K_right, D_right, reproj_error_right)


def to_terminal_image(img_bgr: np.ndarray) -> KittyImage:
    """Convert a BGR OpenCV image to a term-image instance."""
    # Handle grayscale images
    if len(img_bgr.shape) == 2:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return KittyImage(pil_img, width=100)


def numpy_to_ros_imu(timestamp: float, acc: np.ndarray, gyro: np.ndarray):
    """Convert numpy arrays to ROS sensor_msgs/Imu message."""
    # Convert timestamp to ROS Time
    sec = int(timestamp)
    nanosec = int((timestamp - sec) * 1e9)
    
    # Create message object using SimpleNamespace (matches rosbags message structure)
    msg = SimpleNamespace()
    msg.header = SimpleNamespace()
    msg.header.seq = 0  # Sequence number (not critical for bag files)
    msg.header.stamp = SimpleNamespace()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    msg.header.frame_id = "imu"
    msg.orientation = SimpleNamespace()
    msg.orientation.x = 0.0
    msg.orientation.y = 0.0
    msg.orientation.z = 0.0
    msg.orientation.w = 1.0
    msg.orientation_covariance = np.zeros(9, dtype=np.float64)
    msg.angular_velocity = SimpleNamespace()
    msg.angular_velocity.x = float(gyro[0])
    msg.angular_velocity.y = float(gyro[1])
    msg.angular_velocity.z = float(gyro[2])
    msg.angular_velocity_covariance = np.zeros(9, dtype=np.float64)
    msg.linear_acceleration = SimpleNamespace()
    msg.linear_acceleration.x = float(acc[0])
    msg.linear_acceleration.y = float(acc[1])
    msg.linear_acceleration.z = float(acc[2])
    msg.linear_acceleration_covariance = np.zeros(9, dtype=np.float64)
    
    return msg


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="Output bag file path")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--skip", type=int, default=100, help="Skip the first N frames")
    parser.add_argument("--max-frames", type=int, default=2000, help="Stop after this many processed frames (0 = infinite)")
    parser.add_argument("--calibrate", action="store_true", help="Perform camera calibration after recording")
    parser.add_argument("--checkerboard-cols", type=int, default=10, help="Number of inner corners in checkerboard (columns)")
    parser.add_argument("--checkerboard-rows", type=int, default=7, help="Number of inner corners in checkerboard (rows)")
    parser.add_argument("--checkerboard-size", type=float, default=0.025, help="Size of checkerboard square in meters")
    parser.add_argument("--disable-distortion", action="store_true", help="Disable distortion model (set distortion coefficients to zero)")
    parser.add_argument("--no-display", action="store_true", help="Disable terminal image display")
    args = parser.parse_args(argv)

    disable_queries()  # Avoid terminal query round-trips that block in non-interactive sessions.

    display_every = 10 # display every 10 frames

    stream = rs_sdk.D435iIterator(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    try:
        for i in range(args.skip):
            first = next(iter(stream))
    except StopIteration:
        print("No frames received from RealSense. Is the D435i connected?", file=sys.stderr)
        return

    # Initialize bag writer
    typestore = get_typestore(Stores.ROS1_NOETIC)
    
    # Get message definitions (ROS2 style type names work with ROS1_NOETIC typestore)
    image_msgdef, image_md5 = typestore.generate_msgdef("sensor_msgs/msg/Image")
    imu_msgdef, imu_md5 = typestore.generate_msgdef("sensor_msgs/msg/Imu")
    
    # Store images for calibration if requested (must be outside with block)
    calibration_images = [] if args.calibrate else None

    # remove the output file if it exists
    if args.output.exists():
        args.output.unlink()
    
    print(f"Writing to bag file: {args.output}")
    with Writer(args.output) as writer:
        # Add connections for the three topics
        # Note: add_connection expects ROS2-style type names (with /msg/)
        left_conn = writer.add_connection(
            "/left",
            "sensor_msgs/msg/Image",  # ROS2 style type name
            typestore=typestore,
            msgdef=image_msgdef,
            md5sum=image_md5,
        )
        right_conn = writer.add_connection(
            "/right",
            "sensor_msgs/msg/Image",  # ROS2 style type name
            typestore=typestore,
            msgdef=image_msgdef,
            md5sum=image_md5,
        )
        imu_conn = writer.add_connection(
            "/imu",
            "sensor_msgs/msg/Imu",  # ROS2 style type name
            typestore=typestore,
            msgdef=imu_msgdef,
            md5sum=imu_md5,
        )

        print("Recording to bag (Ctrl+C to stop)...")
        frame_idx = 0
        processed_frames = 0
        log_start = time.perf_counter()
        log_count = 0
        display_start = time.perf_counter()
        display_count = 0
        clear = "\033[2J\033[H"

        try:
            for t_curr, left_rect, right_rect, imu_ts, imu_gyro, imu_acc in stream:
                frame_idx += 1

                # Store images for calibration if requested
                if calibration_images is not None:
                    # Store every Nth frame to avoid storing too many
                    if frame_idx % 6 == 0:  # Store every 6th frame
                        calibration_images.append((left_rect.copy(), right_rect.copy()))
                
                # Convert images to ROS Image messages
                left_msg = numpy_to_ros_image(left_rect, float(t_curr), encoding="mono8")
                right_msg = numpy_to_ros_image(right_rect, float(t_curr), encoding="mono8")

                # Serialize and write image messages
                timestamp_ns = int(float(t_curr) * 1e9)
                left_data = typestore.serialize_ros1(left_msg, "sensor_msgs/msg/Image")
                right_data = typestore.serialize_ros1(right_msg, "sensor_msgs/msg/Image")
                writer.write(left_conn, timestamp_ns, left_data)
                writer.write(right_conn, timestamp_ns, right_data)

                # # apply accelerometer calibration
                # acc_M = np.linalg.inv(np.array([
                #     [0.9975482, 0.0, 0.0],
                #     [0.00216291, 0.99217217, 0.0],
                #     [-0.00287838, -0.01121157, 1.00496772]
                # ]))
                # for i in range(imu_acc.shape[0]):
                #     imu_acc[i] = acc_M @ imu_acc[i]

                # Write IMU messages
                if imu_ts.size > 0:
                    for i in range(imu_ts.size):
                        imu_msg = numpy_to_ros_imu(
                            float(imu_ts[i]),
                            imu_acc[i],
                            imu_gyro[i],
                        )
                        imu_timestamp_ns = int(float(imu_ts[i]) * 1e9)
                        imu_data = typestore.serialize_ros1(imu_msg, "sensor_msgs/msg/Imu")
                        writer.write(imu_conn, imu_timestamp_ns, imu_data)

                processed_frames += 1
                log_count += 1
                display_count += 1

                # Display images in terminal if enabled
                if not args.no_display and display_count % display_every == 0:
                    # Convert grayscale to BGR for display
                    if len(left_rect.shape) == 2:
                        left_display = cv2.cvtColor(left_rect, cv2.COLOR_GRAY2BGR)
                    else:
                        left_display = left_rect.copy()
                    
                    if len(right_rect.shape) == 2:
                        right_display = cv2.cvtColor(right_rect, cv2.COLOR_GRAY2BGR)
                    else:
                        right_display = right_rect.copy()
                    
                    # Downscale by factor for faster display
                    scale_factor = 8
                    left_small = cv2.resize(left_display, 
                                           (left_display.shape[1] // scale_factor, 
                                            left_display.shape[0] // scale_factor),
                                           interpolation=cv2.INTER_AREA)
                    right_small = cv2.resize(right_display,
                                             (right_display.shape[1] // scale_factor,
                                              right_display.shape[0] // scale_factor),
                                             interpolation=cv2.INTER_AREA)
                    
                    # Concatenate left and right side-by-side
                    combined = np.hstack([left_small, right_small])
                    
                    # Convert to terminal image (single conversion is faster)
                    term_img = to_terminal_image(combined)
                    
                    now = time.perf_counter()
                    elapsed = now - display_start
                    fps = display_count / elapsed if elapsed > 0 else 0
                    
                    print(clear, end="")
                    print(f"Frame {processed_frames} @ {t_curr:.3f}s | {fps:.2f} Hz | LEFT | RIGHT")
                    print(term_img)
                    
                    # Reset display counters every second for smooth fps calculation
                    if elapsed >= 1.0:
                        display_start = now
                        display_count = 0

                now = time.perf_counter()
                if now - log_start >= 1.0:
                    fps = log_count / (now - log_start)
                    if args.no_display:
                        print(f"Recorded {log_count} frames in the last {now - log_start:.2f}s ({fps:.2f} Hz), total processed {processed_frames}")
                    log_start = now
                    log_count = 0

                if args.max_frames and processed_frames >= args.max_frames:
                    print(f"Reached max_frames={args.max_frames}, stopping stream.")
                    break

        except KeyboardInterrupt:
            print("Keyboard interrupt received, stopping...")
        finally:
            stream.close()
            print(f"Bag file saved to: {args.output}")
    
    # Perform camera calibration if requested
    if args.calibrate and calibration_images and len(calibration_images) > 0:
        print("\n" + "="*80)
        print("CAMERA CALIBRATION")
        print("="*80)
        
        try:
            checkerboard_size = (args.checkerboard_cols, args.checkerboard_rows)
            (K_left, D_left, error_left), (K_right, D_right, error_right) = calibrate_camera(
                calibration_images,
                checkerboard_size,
                args.checkerboard_size,
                disable_distortion=args.disable_distortion,
            )
            
            print("\n" + "-"*80)
            print("LEFT CAMERA INTRINSICS")
            print("-"*80)
            print(f"Camera Matrix K:")
            print(f"  [{K_left[0,0]:.6f}, {K_left[0,1]:.6f}, {K_left[0,2]:.6f}]")
            print(f"  [{K_left[1,0]:.6f}, {K_left[1,1]:.6f}, {K_left[1,2]:.6f}]")
            print(f"  [{K_left[2,0]:.6f}, {K_left[2,1]:.6f}, {K_left[2,2]:.6f}]")
            print(f"\nFocal length: fx={K_left[0,0]:.6f}, fy={K_left[1,1]:.6f}")
            print(f"Principal point: cx={K_left[0,2]:.6f}, cy={K_left[1,2]:.6f}")
            
            if args.disable_distortion:
                print(f"\nDistortion model: DISABLED (all coefficients set to zero)")
                print(f"Distortion coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]")
            else:
                print(f"\nDistortion coefficients (k1, k2, p1, p2, k3):")
                print(f"  [{D_left[0,0]:.6f}, {D_left[1,0]:.6f}, {D_left[2,0]:.6f}, {D_left[3,0]:.6f}, {D_left[4,0]:.6f}]")
            
            print(f"\nRMS Reprojection Error: {error_left:.6f} pixels")
            
            print("\n" + "-"*80)
            print("RIGHT CAMERA INTRINSICS")
            print("-"*80)
            print(f"Camera Matrix K:")
            print(f"  [{K_right[0,0]:.6f}, {K_right[0,1]:.6f}, {K_right[0,2]:.6f}]")
            print(f"  [{K_right[1,0]:.6f}, {K_right[1,1]:.6f}, {K_right[1,2]:.6f}]")
            print(f"  [{K_right[2,0]:.6f}, {K_right[2,1]:.6f}, {K_right[2,2]:.6f}]")
            print(f"\nFocal length: fx={K_right[0,0]:.6f}, fy={K_right[1,1]:.6f}")
            print(f"Principal point: cx={K_right[0,2]:.6f}, cy={K_right[1,2]:.6f}")
            
            if args.disable_distortion:
                print(f"\nDistortion model: DISABLED (all coefficients set to zero)")
                print(f"Distortion coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]")
            else:
                print(f"\nDistortion coefficients (k1, k2, p1, p2, k3):")
                print(f"  [{D_right[0,0]:.6f}, {D_right[1,0]:.6f}, {D_right[2,0]:.6f}, {D_right[3,0]:.6f}, {D_right[4,0]:.6f}]")
            
            print(f"\nRMS Reprojection Error: {error_right:.6f} pixels")
            print("="*80)
            
        except Exception as e:
            print(f"\nERROR: Camera calibration failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

