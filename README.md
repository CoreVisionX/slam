# SLAM

A real-time stereo vision SLAM system with visual odometry, loop closure detection, and bundle adjustment.

## Installation

Install the package using pixi or pip:

```bash
# Using pixi (recommended for development)
pixi install

# Or install from the git repository
pip install git+https://github.com/your-org/slam.git
```

## Live SLAM Demo

After installation, you can run the live SLAM demo with a stereo camera using the `slam-live` command:

```bash
slam-live --calib calib.npz --camera 0
```

### Basic Usage

The simplest setup requires only a calibration file:

```bash
slam-live --calib path/to/calib.npz
```

### Common Options

- `--calib PATH` - Path to stereo calibration file (default: `calib.npz`)
- `--camera INDEX` - OpenCV camera index (default: 0)
- `--width W` - Capture width for stereo stream (default: 1280)
- `--height H` - Capture height for stereo stream (default: 640)
- `--fps FPS` - Requested capture framerate

### Frame Splitting Options

The system expects a side-by-side stereo feed. Configure how to split it:

- `--split-mode MODE` - Splitting mode: `half`, `px`, or `ratio` (default: `half`)
- `--split-px PX` - Split column when using `--split-mode=px`
- `--split-ratio RATIO` - Split ratio (0-1) when using `--split-mode=ratio` (default: 0.5)
- `--swap-halves` - Swap left/right halves after splitting
- `--resize-to-calib` - Resize each half to calibration resolution

### SLAM Configuration

- `--keyframe-translation T` - Translation threshold in meters for keyframes (default: 0.2)
- `--keyframe-rotation R` - Rotation threshold in degrees for keyframes (default: 10.0)
- `--loop-min-inliers N` - Minimum inliers for loop closure (default: 30)
- `--loop-workers N` - Number of loop closure workers

### Visual Odometry Options

- `--matcher TYPE` - Feature matcher: `lighterglue` or `orb` (default: `orb`)
- `--vo-min-matches N` - Minimum matches for VO estimate (default: 30)
- `--vo-min-inliers N` - Minimum PnP inliers for VO estimate (default: 30)
- `--vo-failure-behavior MODE` - Fallback when VO fails: `repeat` or `identity` (default: `repeat`)
- `--no-rectify-inputs` - Disable frontend rectification
- `--no-vo-rectify-inputs` - Disable VO rectification

### Visualization

- `--rerun-tcp HOST:PORT` - Send visualization to remote Rerun viewer
- `--rerun-app-id ID` - Rerun application ID (default: `live-stereo-slam`)
- `--disable-rerun` - Disable Rerun logging entirely

### Advanced Options

- `--disable-huber` - Disable Huber loss on loop closures
- `--sleep SECONDS` - Sleep time between frames (default: 0.0)
- `--frame-limit N` - Limit number of frames to process
- `--max-read-failures N` - Max consecutive camera failures (default: 10)
- `--perf-log-threshold T` - Log performance for non-keyframes exceeding T seconds
- `--log-loop-closures` - Print loop closure info to console

### Example Usage

Run with a custom calibration and enable performance logging:

```bash
slam-live --calib my_stereo_calib.npz \
          --camera 0 \
          --width 1280 \
          --height 640 \
          --matcher lighterglue \
          --keyframe-translation 0.15 \
          --keyframe-rotation 8.0 \
          --log-loop-closures \
          --perf-log-threshold 0.5
```

Run with a remote Rerun viewer:

```bash
slam-live --calib calib.npz \
          --rerun-tcp 127.0.0.1:9876
```

## TODO
- [ ] fix tartanairpy submodule
- [ ] add lightglue submodule with mask patch
