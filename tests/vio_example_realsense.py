import argparse
from pathlib import Path
import sys
import time

import numpy as np
import rerun as rr

from slam.vio import rs_sdk
from slam.vio.mp_runner import AsyncVIO


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vio-config", type=Path, default=Path("config/vio_d435i.yaml"))
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=float, default=1.0, help="Seconds to wait after spinning up the worker")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed frames (0 = infinite)")
    parser.add_argument("--log-every", type=int, default=5, help="Log position estimate every N frames")
    parser.add_argument("--exposure-us", type=float, default=None, help="Exposure time in microseconds (if not provided, auto-exposure is enabled)")
    parser.add_argument("--gain", type=float, default=None, help="Gain (if not provided, auto-exposure is enabled)")
    parser.add_argument("--init-duration", type=float, default=2.0, help="Duration in seconds to collect IMU data for initial gravity and bias estimation")
    args = parser.parse_args(argv)

    # Only pass exposure_us and gain if they are provided
    # If neither is provided, auto-exposure will be enabled
    stream_kwargs = {
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
    }
    if args.exposure_us is not None:
        stream_kwargs["exposure_us"] = args.exposure_us
    if args.gain is not None:
        stream_kwargs["gain"] = args.gain
    
    stream = rs_sdk.D435iIterator(**stream_kwargs)

    # # Collect IMU data for initial gravity and bias estimation
    # print(f"Collecting IMU data for {args.init_duration:.1f}s to estimate initial gravity and gyro bias...")
    # acc_M = np.linalg.inv(np.array([
    #     [0.9975482, 0.0, 0.0],
    #     [0.00216291, 0.99217217, 0.0],
    #     [-0.00287838, -0.01121157, 1.00496772]
    # ]))
    
    # imu_acc_samples = []
    # imu_gyro_samples = []
    # init_start_time = time.perf_counter()
    # init_end_time = init_start_time + args.init_duration
    
    # first = None
    # t0 = None
    # left0 = None
    # right0 = None
    
    # # Collect samples until we have enough data
    # try:
    #     for t_curr, left_rect, right_rect, imu_ts, imu_gyro, imu_acc in stream:
    #         # Store first frame for later use
    #         if first is None:
    #             t0, left0, right0 = t_curr, left_rect.copy(), right_rect.copy()
    #             first = (t_curr, left0, right0, imu_ts, imu_gyro, imu_acc)
            
    #         current_time = time.perf_counter()
    #         if current_time >= init_end_time:
    #             break
            
    #         if imu_ts.size > 0:
    #             for i in range(imu_acc.shape[0]):
    #                 acc_corrected = acc_M @ imu_acc[i]
    #                 imu_acc_samples.append(acc_corrected)
    #                 imu_gyro_samples.append(imu_gyro[i])
    # except StopIteration:
    #     if first is None:
    #         print("No frames received from RealSense. Is the D435i connected?", file=sys.stderr)
    #         return
    
    # # Close the initialization stream and create a new one for the main loop
    # stream.close()
    # stream = rs_sdk.D435iIterator(**stream_kwargs)
    
    # if len(imu_acc_samples) == 0:
    #     print("Warning: No IMU samples collected during initialization. Using default values.", file=sys.stderr)
    #     initial_gravity = np.array([0, 9.80655, 0])
    #     initial_gyro_bias = np.zeros(3)
    #     initial_acc_bias = np.zeros(3)
    # else:
    #     imu_acc_samples = np.array(imu_acc_samples)
    #     imu_gyro_samples = np.array(imu_gyro_samples)
        
    #     # Estimate gravity as mean of accelerometer readings (assuming device is stationary)
    #     initial_gravity = np.mean(imu_acc_samples, axis=0)
    #     gravity_norm = np.linalg.norm(initial_gravity)
    #     print(f"Estimated gravity: {initial_gravity} (norm: {gravity_norm:.3f})")
        
    #     # Estimate gyro bias as mean of gyroscope readings
    #     initial_gyro_bias = np.mean(imu_gyro_samples, axis=0)
    #     print(f"Estimated gyro bias: {initial_gyro_bias}")
        
    #     # Estimate acc bias (gravity should be the only acceleration when stationary)
    #     # For now, we'll assume acc bias is zero, but we could estimate it if we know the expected gravity direction
    #     initial_acc_bias = np.zeros(3)

    # Estimated gravity: [-0.20434598 -9.79474932 -0.18721172] (norm: 9.799)
    # Estimated gyro bias: [-0.00705335 -0.00172475 -0.00137872]
    initial_gravity = np.array([-0.20434598, -9.79474932, -0.18721172])
    initial_gyro_bias = np.array([-0.00705335, -0.00172475, -0.00137872])
    initial_acc_bias = np.array([0.0, 0.0, 0.0])
    
    # Create AsyncVIO with initial gravity and bias
    async_vio = AsyncVIO(
        vio_config_path=args.vio_config,
        # initial_gravity=initial_gravity,
        # initial_bias_acc=initial_acc_bias,
        # initial_bias_gyro=initial_gyro_bias,
    )

    if args.warmup > 0:
        print(f"Waiting {args.warmup:.1f}s for VIO worker to spin up...")
        time.sleep(args.warmup)

    try:
        # skip first 100 frames since the imu data is a little noisy at the start for some reason. probably add some health checks logging (rerun text logs esp.) to alert users
        for i in range(100):
            first = next(iter(stream))
    except StopIteration:
        raise RuntimeError("No frames received from RealSense. Is the D435i connected?")
    t0, left0, right0, _, _, _ = first

    async_vio.reset(
        timestamp=float(t0),
        left_rect=left0,
        right_rect=right0,
        t=np.zeros(3, dtype=np.float64),
        R=np.eye(3, dtype=np.float64),
        v=np.zeros(3, dtype=np.float64),
    )

    print("Running AsyncVIO (Ctrl+C to stop)...")
    frame_idx = 0
    processed_frames = 0
    log_start = time.perf_counter()
    log_count = 0

    acc_M = np.linalg.inv(np.array([
        [0.9975482, 0.0, 0.0],
        [0.00216291, 0.99217217, 0.0],
        [-0.00287838, -0.01121157, 1.00496772]
    ]))

    try:
        for t_curr, left_rect, right_rect, imu_ts, imu_gyro, imu_acc in stream:
            frame_idx += 1

            if imu_ts.size > 0:
                # for i in range(imu_acc.shape[0]):
                #     imu_acc[i] = acc_M @ imu_acc[i]

                # async_vio.process_imu(imu_ts=imu_ts, imu_acc=imu_acc, imu_gyro=imu_gyro)

                for t_imu, acc, gyro in zip(imu_ts, imu_acc, imu_gyro):
                    # acc = acc_M @ acc

                    async_vio.process_imu(imu_ts=np.array([t_imu]), imu_acc=np.array([acc]), imu_gyro=np.array([gyro]))

            async_vio.process(timestamp=float(t_curr), left_rect=left_rect, right_rect=right_rect)
            processed_frames += 1
            log_count += 1

            now = time.perf_counter()
            if now - log_start >= 1.0:
                fps = log_count / (now - log_start)
                if rr.get_global_data_recording() is not None:
                    rr.log("async_vio/logs", rr.TextLog(f"Enqueued {log_count} frames in the last {now - log_start:.2f}s ({fps:.2f} Hz), total processed {processed_frames}"))
                log_start = now
                log_count = 0

            if args.max_frames and processed_frames >= args.max_frames:
                if rr.get_global_data_recording() is not None:
                    rr.log("async_vio/logs", rr.TextLog(f"Reached max_frames={args.max_frames}, stopping stream."))
                break

            if frame_idx % args.log_every == 0:
                estimate = async_vio.get_current_estimate()
                if estimate is not None:
                    print(f"Estimated Position: {estimate.t}")

    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping...")
    finally:
        stream.close()
        async_vio.stop()


if __name__ == "__main__":
    main()
