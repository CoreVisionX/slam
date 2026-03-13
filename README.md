# CVX Stereo Visual-Inertial Odometry

Stereo VIO based on sparse feature tracking and incremental fixed-lag nonlinear optimization. Targets embedded platforms (Jetson Nano, Raspberry Pi).

https://github.com/user-attachments/assets/02f5a774-cdd6-4d27-bb30-3372ba7a7746

## Overview

The front end tracks sparse features across stereo frames using KLT optical flow and recovers per-feature depth through template-based stereo matching, both implemented in C++ for throughput. Inter-frame motion is initialized via PnP-RANSAC on tracked 3D-2D correspondences, with an IMU-predicted constant-velocity fallback when tracking degrades. Between keyframes, accelerometer and gyroscope measurements are accumulated into preintegrated IMU factors. The back end is a keyframe-based incremental fixed-lag smoother (GTSAM's iSAM2) that jointly optimizes over poses, velocities, IMU biases, and 3D landmarks within a sliding window. Outlier observations are rejected through a combination of RANSAC inlier gating, reprojection error thresholding, Huber robust costs, and median filtering. An optional ORB-based relocalization module provides drift-bounded loop closures back to the origin frame.

## EuRoC Results

ATE RMSE (meters) on the [EuRoC MAV dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets), SE(3) Umeyama alignment. Comparison against published stereo results from [Basalt](https://arxiv.org/abs/1904.06504), [OpenVINS](https://arxiv.org/abs/1910.01122), and [Kimera](https://arxiv.org/abs/1910.02490):

| Sequence | Ours | Basalt | OpenVINS | Kimera |
|---|---|---|---|---|
| MH_01_easy | 0.11 | 0.07 | 0.07 | 0.11 |
| MH_02_easy | 0.06 | 0.06 | 0.14 | 0.10 |
| MH_03_medium | 0.11 | 0.07 | 0.09 | 0.16 |
| MH_04_difficult | 0.09 | 0.13 | 0.17 | 0.24 |
| MH_05_difficult | 0.14 | 0.11 | 0.25 | 0.35 |
| V1_01_easy | 0.05 | 0.04 | 0.06 | 0.05 |
| V1_02_medium | 0.07 | 0.05 | 0.06 | 0.08 |
| V1_03_difficult | 0.12 | 0.10 | 0.06 | 0.07 |
| V2_01_easy | 0.05 | 0.04 | 0.05 | 0.08 |
| V2_02_medium | 0.08 | 0.05 | 0.05 | 0.10 |
| **Average** | **0.09** | **0.07** | **0.10** | **0.13** |

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

Docker images are provided for Jetson (JetPack 6) and Ubuntu ARM64. Both include librealsense, GTSAM, OpenCV, and all Python dependencies.

```bash
# Jetson (aarch64, JetPack 6)
docker build -f docker/Dockerfile.jetson -t slam .

# Ubuntu ARM64
docker build -f docker/Dockerfile.ubuntu-arm64 -t slam .

# Run with RealSense USB access
docker run -it --privileged -v /dev/bus/usb:/dev/bus/usb slam pixi shell
```

## Dependencies

GTSAM, OpenCV, pybind11, Rerun, Hydra
