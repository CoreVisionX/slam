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
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--warmup", type=float, default=1.0, help="Seconds to wait after spinning up the worker")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed frames (0 = infinite)")
    args = parser.parse_args(argv)

    async_vio = AsyncVIO(vio_config_path=args.vio_config)

    if args.warmup > 0:
        print(f"Waiting {args.warmup:.1f}s for VIO worker to spin up...")
        time.sleep(args.warmup)

    stream = rs_sdk.D435iIterator(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    try:
        first = next(iter(stream))
    except StopIteration:
        print("No frames received from RealSense. Is the D435i connected?", file=sys.stderr)
        return

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

    try:
        for t_curr, left_rect, right_rect, imu_ts, imu_gyro, imu_acc in stream:
            frame_idx += 1

            if imu_ts.size > 0:
                async_vio.process_imu(imu_ts=imu_ts, imu_acc=imu_acc, imu_gyro=imu_gyro)

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

    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping...")
    finally:
        stream.close()
        async_vio.stop()


if __name__ == "__main__":
    main()
