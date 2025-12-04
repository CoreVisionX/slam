# Building Wheels

## Prerequisites

Ensure the pixi environment is set up:
```bash
pixi install
```

## Build Commands

```bash
# Standard build
PATH=".pixi/envs/default/bin:$PATH" uv build --wheel

# Platform variant builds (Jetson, RPi, etc.)
SLAM_BUILD_VARIANT=jetson PATH=".pixi/envs/default/bin:$PATH" uv build --wheel
SLAM_BUILD_VARIANT=rpi PATH=".pixi/envs/default/bin:$PATH" uv build --wheel
```

Output wheels land in `dist/`:
- `slam-0.1.0-cp311-cp311-linux_aarch64.whl`
- `slam-0.1.0+jetson-cp311-cp311-linux_aarch64.whl`
- `slam-0.1.0+rpi-cp311-cp311-linux_aarch64.whl`

## Install

```bash
pip install dist/slam-0.1.0+jetson-cp311-cp311-linux_aarch64.whl
# or
uv pip install dist/slam-0.1.0+jetson-cp311-cp311-linux_aarch64.whl
```

