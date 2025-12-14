import argparse
from pathlib import Path
import sys
import time

import numpy as np
import rerun as rr

from tqdm import tqdm

from slam.vio import rs_sdk

IMU_RATE = 200.0 # Hz

def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=20_000, help="Stop after this many processed samples")
    parser.add_argument("--rerun-url", type=str, help="Rerun URL")
    args = parser.parse_args(argv)

    stream = rs_sdk.D435iIterator()
    rr.init("realsense_estimate_gravity")
    rr.connect_grpc(args.rerun_url)

    # gather IMU samples
    try:
        imu_ts_samples = []
        imu_acc_samples = []
        imu_gyro_samples = []

        imu_dts_samples = []

        with tqdm(total=args.max_samples) as pbar:
            finished = False

            for _t_curr, _left_rect, _right_rect, imu_ts, imu_gyro, imu_acc in stream:
                for t, acc, gyro in zip(imu_ts, imu_acc, imu_gyro):
                    imu_ts_samples.append(t)
                    imu_acc_samples.append(acc)
                    imu_gyro_samples.append(gyro)

                    if len(imu_ts_samples) == 1:
                        dt = 0.0
                    else:
                        dt = t - imu_ts_samples[-2]
                    imu_dts_samples.append(dt)

                    rr.set_time("imu", timestamp=t)
                    rr.log("acc/x", rr.Scalars(acc[0]))
                    rr.log("acc/y", rr.Scalars(acc[1]))
                    rr.log("acc/z", rr.Scalars(acc[2]))
                    rr.log("gyro/x", rr.Scalars(gyro[0]))
                    rr.log("gyro/y", rr.Scalars(gyro[1]))
                    rr.log("gyro/z", rr.Scalars(gyro[2]))
                    rr.log("dts", rr.Scalars(dt))

                    pbar.update(1)
                    if len(imu_ts_samples) >= args.max_samples:
                        finished = True
                        break

                if finished:
                    break

    except Exception as err:
        raise err
    finally:
        stream.close()

    imu_ts_samples = np.array(imu_ts_samples)
    imu_acc_samples = np.array(imu_acc_samples)
    imu_gyro_samples = np.array(imu_gyro_samples)
    imu_dts_samples = np.array(imu_dts_samples)

    # estimate gravity
    gravity = np.mean(imu_acc_samples, axis=0)
    gravity_norm = np.linalg.norm(gravity)
    gravity_std = np.std(imu_acc_samples, axis=0)

    print(f"Gravity: {gravity}")
    print(f"Gravity std: {gravity_std}")
    print(f"Gravity norm: {gravity_norm}")

    # estimate gyro bias
    gyro_bias = np.mean(imu_gyro_samples, axis=0)
    gyro_bias_std = np.std(imu_gyro_samples, axis=0)
    print(f"Gyro bias: {gyro_bias}")
    print(f"Gyro bias std: {gyro_bias_std}")
    print(f"Gyro bias norm: {np.linalg.norm(gyro_bias)}")

    # estimate noise characteristics
    dt = 1.0 / IMU_RATE
    acc_noise = gravity_std * np.sqrt(dt) # convert to m/s^2 / sqrt(Hz)
    gyro_noise = gyro_bias_std * np.sqrt(dt) # convert to rad/s / sqrt(Hz)
    print(f"Acc noise: {acc_noise * 10}")
    print(f"Gyro noise: {gyro_noise * 10}")

    
    

if __name__ == "__main__":
    main()

