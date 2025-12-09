import argparse
from pathlib import Path
import time

import numpy as np

from slam.hydra_utils import compose_config
from slam.vio import rs_sdk
from slam.vio.mp_runner import AsyncVIO


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vio-config", type=Path, default=Path("config/vio_d435i.yaml"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=float, default=1.0, help="Seconds to wait after spinning up the worker")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed frames (0 = infinite)")
    parser.add_argument("--log-every", type=int, default=60, help="Log position estimate every N frames")
    parser.add_argument("--init-duration", type=float, default=2.0, help="Duration in seconds to collect IMU data for initial gravity and bias estimation")
    parser.add_argument("--skip-frames", type=int, default=100, help="Skip the first N frames")
    args = parser.parse_args(argv)

    config = compose_config(args.vio_config)
    width = config.config.width
    height = config.config.height
    
    stream = rs_sdk.D435iIterator(width=width, height=height, fps=args.fps)

    # Use the first few frames to estimate gravity and gyro bias
    init_frames = 60
    imu_ts_samples = []
    imu_acc_samples = []
    imu_gyro_samples = []

    for i, (t_curr, left_rect, right_rect, imu_ts, imu_gyro, imu_acc) in enumerate(stream):
        if imu_ts.size > 0:
            for t_imu, acc, gyro in zip(imu_ts, imu_acc, imu_gyro):
                imu_ts_samples.append(t_imu)
                imu_acc_samples.append(acc)
                imu_gyro_samples.append(gyro)

        if i >= init_frames:
            break

    imu_ts_samples = np.array(imu_ts_samples)
    imu_acc_samples = np.array(imu_acc_samples)
    imu_gyro_samples = np.array(imu_gyro_samples)

    # Estimate gravity and gyro bias
    gravity = -np.mean(imu_acc_samples, axis=0)
    gravity_norm = np.linalg.norm(gravity)
    gyro_bias = np.mean(imu_gyro_samples, axis=0)
    print(f"Gravity: {gravity}")
    print(f"Gravity norm: {gravity_norm}")
    print(f"Gyro bias: {gyro_bias}")

    # Create AsyncVIO with initial gravity and bias via overrides
    async_vio = AsyncVIO(
        vio_config_path=args.vio_config,
        vio_overrides=[
            "imu_preintegrator.config.gravity=" + str(gravity.tolist()),
            "imu_preintegrator.config.initial_gyro_bias=" + str(gyro_bias.tolist())
        ]
    )

    if args.warmup > 0:
        print(f"Waiting {args.warmup:.1f}s for VIO worker to spin up...")
        time.sleep(args.warmup)

    try:
        # skip first N frames, sometimes the first few frames of data aren't synced properly
        for i in range(args.skip_frames):
            next(iter(stream))
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
    processed_frames = 0

    try:
        for t_curr, left_rect, right_rect, imu_ts, imu_gyro, imu_acc in stream:
            async_vio.process_imu(imu_ts=imu_ts, imu_acc=imu_acc, imu_gyro=imu_gyro)
            async_vio.process(timestamp=float(t_curr), left_rect=left_rect, right_rect=right_rect)

            processed_frames += 1

            if processed_frames % args.log_every == 0:
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
