# CVX Stereo Visual-Inertial Odometry

Stereo VIO based on sparse feature tracking and incremental fixed-lag nonlinear optimization. Targets embedded platforms (Jetson Nano, Raspberry Pi).

https://github.com/user-attachments/assets/02f5a774-cdd6-4d27-bb30-3372ba7a7746

## Overview

The front end tracks sparse features across stereo frames using KLT optical flow and recovers per-feature depth through template-based stereo matching, both implemented in C++ for throughput. Inter-frame motion is initialized via PnP-RANSAC on tracked 3D-2D correspondences, with an IMU-predicted constant-velocity fallback when tracking degrades. Between keyframes, accelerometer and gyroscope measurements are accumulated into preintegrated IMU factors. The back end is a keyframe-based incremental fixed-lag smoother (GTSAM's iSAM2) that jointly optimizes over poses, velocities, IMU biases, and 3D landmarks within a sliding window. Outlier observations are rejected through a combination of RANSAC inlier gating, reprojection error thresholding, Huber robust costs, and median filtering. An optional ORB-based relocalization module provides drift-bounded loop closures back to the origin frame.

## EuRoC Results

ATE RMSE (meters) on the [EuRoC MAV dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets), SE(3) Umeyama alignment:

| Sequence        | Distance (m) | ATE RMSE (m) | % of distance |
| --------------- | ------------ | ------------ | ------------- |
| V1_01_easy      | 57.8         | 0.05         | 0.09%         |
| V2_01_easy      | 35.8         | 0.05         | 0.15%         |
| MH_02_easy      | 73.2         | 0.06         | 0.08%         |
| V1_02_medium    | 75.1         | 0.07         | 0.09%         |
| V2_02_medium    | 82.4         | 0.08         | 0.10%         |
| MH_04_difficult | 91.3         | 0.09         | 0.10%         |
| MH_03_medium    | 130.9        | 0.11         | 0.09%         |
| MH_01_easy      | 80.4         | 0.11         | 0.14%         |
| V1_03_difficult | 77.8         | 0.12         | 0.15%         |
| MH_05_difficult | 97.6         | 0.14         | 0.15%         |
| **Average**     | **80.2**     | **0.09**     | **0.11%**     |

## Usage

```python
vio = VIO.from_config("vio_d435i.yaml")
vio.reset(timestamp, left, right, t=t0, R=R0)

estimate = vio.process(
    timestamp=ts,
    left_rect=left, right_rect=right,
    imu_acc=acc, imu_gyro=gyro, imu_ts=imu_ts,
)
# estimate.t, estimate.R, estimate.v
```

## Building

See [BUILDING.md](BUILDING.md) and [INSTALL.md](INSTALL.md). Alternatively, Docker images provided for Jetson and Ubuntu ARM64.

## Dependencies

GTSAM, OpenCV, pybind11, Rerun, Hydra
