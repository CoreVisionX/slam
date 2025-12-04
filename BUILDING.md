# Building Wheels

## Prerequisites

Ensure the pixi environment is set up:
```bash
pixi install
```

Make sure OpenCV is installed on your system.

## Build Commands

```bash
# Jetson build
pixi run wheel-jetson
```

See build commands for additional platforms in `pixi.toml`.

Output wheels land in `dist/`:
- `slam-0.1.0-cp311-cp311-manylinux_2_35_aarch64.whl`
- `slam-0.1.0+jetson-cp311-cp311-manylinux_2_35_aarch64.whl`
- `slam-0.1.0+rpi-cp311-cp311-manylinux_2_35_aarch64.whl`

## Install

```bash
pip install slam-0.1.0+jetson-cp311-cp311-manylinux_2_35_aarch64.whl
# or
uv add slam-0.1.0+jetson-cp311-cp311-manylinux_2_35_aarch64.whl
```

