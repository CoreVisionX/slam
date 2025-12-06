import argparse
from pathlib import Path
import sys
import time

import numpy as np
import rerun as rr

from tqdm import tqdm

from slam.vio import rs_sdk


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=1_000, help="Stop after this many processed samples")
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

            gyro_M = np.array([
                [1.99108148, 0.0, 0.0],
                [-0.00233388, 2.00310951, 0.0],
                [-0.00313549, -0.00798199, 2.00860319]
            ])
            gyro_A = np.array([
                [0.00003872, -0.00054633, 0.00009226],
                [0.00026314, 0.00061978, -0.0001102],
                [0.00002499, 0.00002004, -0.0000637]
            ])
            acc_M = np.array([
                [0.99884124, 0.0, 0.0],
                [-0.00879978, 0.99821107, 0.0],
                [-0.00942307, -0.00389955, 0.99772294]
            ])

            gyro_M_inv = np.linalg.inv(gyro_M)
            acc_M_inv = np.linalg.inv(acc_M)

            def correct_acc(*, raw_acc: np.ndarray) -> np.ndarray:
                return acc_M_inv @ raw_acc
            
            def correct_gyro(*, raw_gyro: np.ndarray, corrected_acc: np.ndarray) -> np.ndarray:
                return gyro_M_inv @ (raw_gyro - gyro_A @ corrected_acc)


            for _t_curr, _left_rect, _right_rect, imu_ts, imu_gyro, imu_acc in stream:
                for t, acc, gyro in zip(imu_ts, imu_acc, imu_gyro):
                    corrected_acc = correct_acc(raw_acc=acc)
                    corrected_gyro = correct_gyro(raw_gyro=gyro, corrected_acc=corrected_acc)

                    imu_ts_samples.append(t)
                    imu_acc_samples.append(corrected_acc)
                    imu_gyro_samples.append(corrected_gyro)

                    if len(imu_ts_samples) == 1:
                        dt = 0.0
                    else:
                        dt = t - imu_ts_samples[-2]
                    imu_dts_samples.append(dt)

                    rr.set_time("imu", timestamp=t)
                    rr.log("acc/x", rr.Scalars(corrected_acc[0]))
                    rr.log("acc/y", rr.Scalars(corrected_acc[1]))
                    rr.log("acc/z", rr.Scalars(corrected_acc[2]))
                    rr.log("gyro/x", rr.Scalars(corrected_gyro[0]))
                    rr.log("gyro/y", rr.Scalars(corrected_gyro[1]))
                    rr.log("gyro/z", rr.Scalars(corrected_gyro[2]))
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
    
    
    

if __name__ == "__main__":
    main()

