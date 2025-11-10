#!/usr/bin/env python3
"""
stereo_calibrate.py
A small utility to:
  1) Capture stereo chessboard samples from a single "wide" UVC frame (left|right halves).
  2) Calibrate intrinsics per eye and stereo extrinsics (R, T, E, F).
  3) Compute rectification maps (R1,R2,P1,P2,Q) and preview rectified video.
  4) Optionally generate a printable checkerboard target (PDF/PNG).

Assumptions:
- Your stereo device outputs one frame where the left half is "left camera" and the right half is "right camera".
- Standard pinhole model with 5 distortion coeffs (k1,k2,p1,p2,k3). For fisheye lenses, this script is not configured.
- Default pattern is a chessboard with inner corners (cols x rows). E.g., 9x6 inner corners is a common choice.

Example usage:
  # 1) Generate a 9x6 inner-corners board, 30mm squares (PDF)
  uv run stereo_calibrate.py make-board --cols 9 --rows 6 --square-mm 24 --out checkerboard_9x6_24mm.pdf

  # 2) Capture samples (watch for detected corners on both views, press SPACE to save a sample, 'q' to quit)
  uv run stereo_calibrate.py capture --out data.npz --camera 0 --cols 9 --rows 6 --square 0.024 --split half

  # 3) Calibrate from samples (data.npz) and write calibration (calib.npz)
  uv run stereo_calibrate.py calibrate --data data.npz --out calib.npz

  # 4) Preview live rectification using the saved calibration
  uv run stereo_calibrate.py preview --calib calib.npz --camera 0 --split half

  # 5) Preview stereo odometry using the saved calibration
  uv run stereo_calibrate.py preview-odometry

Notes on units:
- --square is in **meters** (0.03 for 30 mm); this sets the real-world scale so your baseline has the right units.
- The baseline magnitude is ||T|| where T is the translation from LEFT to RIGHT camera coordinates.
"""

import argparse
import sys
import time
import math
import os
import json
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

import numpy as np
import cv2

try:
    import gtsam
    from gtsam import Pose3, Rot3, Point3, NonlinearFactorGraph, Values, ISAM2
    try:
        from gtsam.symbol_shorthand import X
    except Exception:
        def X(i: int) -> int:
            return gtsam.symbol('x', i)
    GTSAM_OK = True
except Exception as exc:
    print("Loop-closure requires GTSAM (pip install gtsam). Disabling.", exc)
    gtsam = None
    Pose3 = Rot3 = Point3 = NonlinearFactorGraph = Values = ISAM2 = None

    def X(i: int) -> int:  # pragma: no cover - should never be used without GTSAM
        raise RuntimeError("GTSAM is unavailable; loop-closure disabled.")

    GTSAM_OK = False

# ---------------------------- Utilities ----------------------------

@dataclass
class PatternSpec:
    cols: int            # number of inner corners along X (columns)
    rows: int            # number of inner corners along Y (rows)
    square_size: float   # meters per grid cell (square size)

def split_left_right(frame, mode="half", split_px=None, ratio=0.5, swap=False):
    """Split a wide frame into left/right images.
    mode: 'half' (split at width/2), 'px' (split at given column), or 'ratio' (0..1).
    """
    h, w = frame.shape[:2]
    if mode == "half":
        mid = w // 2
    elif mode == "px":
        if split_px is None:
            raise ValueError("split_px must be provided when mode='px'")
        mid = int(split_px)
    elif mode == "ratio":
        mid = int(w * float(ratio))
    else:
        raise ValueError("Unknown split mode")

    left = frame[:, :mid]
    right = frame[:, mid:]
    if swap:
        left, right = right, left
    return left, right

def make_object_points(spec: PatternSpec):
    # OpenCV expects patternSize=(cols, rows) and object points laid out accordingly
    objp = np.zeros((spec.rows * spec.cols, 3), np.float32)
    grid = np.mgrid[0:spec.cols, 0:spec.rows].T.reshape(-1, 2)  # (cols, rows) order
    objp[:, :2] = grid * spec.square_size
    return objp

def find_corners(gray, pattern_size, use_sb=True):
    """Find chessboard corners. If available, try the more robust SB detector."""
    if use_sb and hasattr(cv2, 'findChessboardCornersSB'):
        # SB flags are typically more robust for poor lighting or slight occlusion
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
        if found and corners is not None:
            corners = corners.astype(np.float32).reshape(-1, 1, 2)

            # refine to subpixel
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return found, corners
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if found:
            # refine to subpixel
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return found, corners

def draw_info(img, lines):
    y = 20
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        y += 18

def compute_reprojection_error(objpoints, imgpoints, K, D, rvecs, tvecs):
    total_err = 0.0
    total_points = 0
    per_view_errors = []
    for i, objp in enumerate(objpoints):
        imgp2, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, D)
        err = cv2.norm(imgpoints[i], imgp2, cv2.NORM_L2)
        n = len(objp)
        per_view_errors.append(math.sqrt(err*err / n))
        total_err += err*err
        total_points += n
    rms = math.sqrt(total_err / total_points) if total_points > 0 else float('inf')
    return rms, per_view_errors

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion in (x, y, z, w) order."""
    trace = np.trace(R)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
            w = (R[2,1] - R[1,2]) / s
        elif R[1,1] > R[2,2]:
            s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
            w = (R[0,2] - R[2,0]) / s
        else:
            s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
            w = (R[1,0] - R[0,1]) / s
    return np.array([x, y, z, w], dtype=np.float32)

def gather_depth_samples(points: np.ndarray, depth_map: np.ndarray, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Filter 2D points by valid depth and return corresponding 3D samples."""
    if points is None or len(points) == 0:
        return np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    h, w = depth_map.shape
    valid_pixels = []
    valid_points3d = []
    for pt in points.reshape(-1, 2):
        u = int(round(float(pt[0])))
        v = int(round(float(pt[1])))
        if u < 0 or u >= w or v < 0 or v >= h:
            continue
        z = depth_map[v, u]
        if not math.isfinite(float(z)) or z <= 0.0:
            continue
        valid_pixels.append([pt[0], pt[1]])
        valid_points3d.append(points_3d[v, u])
    if not valid_pixels:
        return np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    pts2d = np.asarray(valid_pixels, dtype=np.float32).reshape(-1, 1, 2)
    pts3d = np.asarray(valid_points3d, dtype=np.float32)
    return pts2d, pts3d


METRIC3D_MEAN_BGR = np.array([123.675, 116.28, 103.53], dtype=np.float32)
METRIC3D_STD_BGR = np.array([58.395, 57.12, 57.375], dtype=np.float32)
METRIC3D_MODEL_INPUT_SIZES: Dict[str, Tuple[int, int]] = {
    # (height, width)
    "metric3d_convnext_tiny": (544, 1216),
    "metric3d_convnext_large": (544, 1216),
    "metric3d_vit_small": (616, 1064),
    "metric3d_vit_large": (616, 1064),
    "metric3d_vit_giant2": (616, 1064),
}

_METRIC3D_RUNNERS: Dict[Tuple[str, str], "Metric3DRunner"] = {}


def ensure_metric3d_dependencies() -> None:
    missing: List[str] = []
    try:
        import timm  # type: ignore
        _ = getattr(timm, "__version__", None)
    except Exception:
        missing.append("timm>=0.9")
    try:
        import mmengine  # type: ignore
        _ = getattr(mmengine, "__version__", None)
    except Exception:
        missing.append("mmengine>=0.10")

    has_mmcv = False
    for candidate in ("mmcv", "mmcv_full", "mmcv_lite"):
        try:
            module = __import__(candidate)
            _ = getattr(module, "__version__", None)
            has_mmcv = True
            break
        except Exception:
            continue
    if not has_mmcv:
        missing.append("mmcv-lite>=2.0 (or mmcv)")

    if missing:
        raise RuntimeError(
            "Metric3D requires additional Python packages. "
            f"Install the following before using --depth-engine=metric3d: {', '.join(missing)}"
        )


class Metric3DRunner:
    """Thin wrapper around the Metric3Dv2 torch.hub interface."""

    def __init__(self, model_name: str, device: str, torch_module):
        if model_name not in METRIC3D_MODEL_INPUT_SIZES:
            raise ValueError(f"Unsupported Metric3D model '{model_name}'. Options: {sorted(METRIC3D_MODEL_INPUT_SIZES)}")
        self.model_name = model_name
        self.device = torch_module.device(device)
        self.torch = torch_module
        ensure_metric3d_dependencies()
        try:
            self.model = torch_module.hub.load("YvanYin/Metric3D", model_name, pretrain=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Metric3D model '{model_name}' from torch hub: {exc}") from exc
        self.model = self.model.to(self.device).eval()
        self.input_hw = METRIC3D_MODEL_INPUT_SIZES[model_name]
        mean = torch_module.tensor(METRIC3D_MEAN_BGR, device=self.device, dtype=torch_module.float32)
        std = torch_module.tensor(METRIC3D_STD_BGR, device=self.device, dtype=torch_module.float32)
        self.mean = mean.view(1, 3, 1, 1)
        self.std = std.view(1, 3, 1, 1)
        self._pad_color = METRIC3D_MEAN_BGR.tolist()

    def _prepare_input(self, bgr: np.ndarray) -> Tuple["torch.Tensor", Dict[str, float]]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        in_h, in_w = self.input_hw
        h, w = rgb.shape[:2]
        scale = min(in_h / h, in_w / w)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = in_h - new_h
        pad_w = in_w - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=self._pad_color,
        )
        tensor = self.torch.from_numpy(padded.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std
        meta = {
            "scale": scale,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
            "orig_shape": (h, w),
        }
        return tensor, meta

    def infer_depth(self, bgr: np.ndarray, cam_matrix: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run Metric3D inference on a rectified BGR frame.

        Returns depth map in meters (np.float32) and optional confidence map.
        """
        tensor, meta = self._prepare_input(bgr)
        torch = self.torch
        inference_ctx = getattr(torch, "inference_mode", torch.no_grad)
        with inference_ctx():
            outputs = self.model.inference({"input": tensor})

        if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
            raise RuntimeError("Metric3D inference returned unexpected output format.")

        pred_depth = outputs[0]
        confidence = outputs[1] if len(outputs) > 1 else None
        extra = outputs[2] if len(outputs) > 2 else {}

        # Some versions return (pred_depth, confidence, extra); others mimic tuple.
        if isinstance(pred_depth, dict):
            depth_tensor = pred_depth.get("pred_depth") or pred_depth.get("prediction_depth")
            confidence = pred_depth.get("confidence", confidence)
        else:
            depth_tensor = pred_depth

        if depth_tensor is None:
            raise RuntimeError("Metric3D did not return a depth tensor.")

        depth_tensor = depth_tensor.squeeze()
        if depth_tensor.ndim != 2:
            depth_tensor = depth_tensor.squeeze(0)
        if depth_tensor.ndim != 2:
            raise RuntimeError(f"Metric3D depth tensor has unexpected shape {depth_tensor.shape}")

        pad_top = int(meta["pad_top"])
        pad_bottom = int(meta["pad_bottom"])
        pad_left = int(meta["pad_left"])
        pad_right = int(meta["pad_right"])

        h_pad, w_pad = depth_tensor.shape
        h_end = h_pad - pad_bottom if pad_bottom > 0 else h_pad
        w_end = w_pad - pad_right if pad_right > 0 else w_pad
        depth_tensor = depth_tensor[pad_top:h_end, pad_left:w_end]

        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
        depth_tensor = torch.nn.functional.interpolate(
            depth_tensor,
            size=meta["orig_shape"],
            mode="bilinear",
            align_corners=False,
        )

        fx = float(cam_matrix[0, 0])
        scale = float(meta["scale"])
        canonical_to_real = (fx * scale) / 1000.0
        depth_tensor = depth_tensor * canonical_to_real
        depth_map = depth_tensor.squeeze().detach().cpu().numpy().astype(np.float32)

        conf_map = None
        conf_source = confidence
        if conf_source is None and isinstance(extra, dict):
            conf_source = extra.get("confidence")
        if conf_source is not None:
            conf_tensor = conf_source.squeeze()
            if conf_tensor.ndim != 2:
                conf_tensor = conf_tensor.squeeze(0)
            if conf_tensor.ndim == 2:
                conf_tensor = conf_tensor[pad_top:h_end, pad_left:w_end]
                conf_tensor = conf_tensor.unsqueeze(0).unsqueeze(0)
                conf_tensor = torch.nn.functional.interpolate(
                    conf_tensor,
                    size=meta["orig_shape"],
                    mode="bilinear",
                    align_corners=False,
                )
                conf_map = conf_tensor.squeeze().detach().cpu().numpy().astype(np.float32)

        return depth_map, conf_map


def get_metric3d_runner(model_name: str, device: Optional[str] = None) -> Metric3DRunner:
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Metric3D requires PyTorch. Install torch before using --depth-engine=metric3d.") from exc

    if device is None:
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "mps"
    key = (model_name, device)
    runner = _METRIC3D_RUNNERS.get(key)
    if runner is None:
        runner = Metric3DRunner(model_name, device, torch)
        _METRIC3D_RUNNERS[key] = runner
    return runner


def compute_pixel_rays(image_shape: Tuple[int, int], cam_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute normalized pixel coordinates (x/z, y/z) for each pixel."""
    h, w = image_shape
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    fx = float(cam_matrix[0, 0])
    fy = float(cam_matrix[1, 1])
    cx = float(cam_matrix[0, 2])
    cy = float(cam_matrix[1, 2])
    x_norm = (grid_x - cx) / fx
    y_norm = (grid_y - cy) / fy
    return x_norm, y_norm


def depth_to_points_from_rays(depth_map: np.ndarray, x_norm: np.ndarray, y_norm: np.ndarray) -> np.ndarray:
    """Convert depth + precomputed rays into XYZ coordinates."""
    points = np.stack(
        (
            x_norm * depth_map,
            y_norm * depth_map,
            depth_map,
        ),
        axis=-1,
    )
    return points.astype(np.float32, copy=False)


def Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Convert rotation+translation into 4x4 transform."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def T_to_Rt(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split 4x4 transform into rotation and translation components."""
    return T[:3, :3].copy(), T[:3, 3].copy()


def compose(Ta: np.ndarray, Tb: np.ndarray) -> np.ndarray:
    """Compose SE3 transforms (Ta * Tb)."""
    return Ta @ Tb


def inv(T: np.ndarray) -> np.ndarray:
    """Invert an SE3 transform."""
    Ri = T[:3, :3].T
    ti = -Ri @ T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = Ri
    Ti[:3, 3] = ti
    return Ti


def pose3_from_Rt(R: np.ndarray, t: np.ndarray) -> "Pose3":
    if not GTSAM_OK:
        raise RuntimeError("pose3_from_Rt requires GTSAM")
    return Pose3(Rot3(np.asarray(R, dtype=np.float64)),
                 Point3(float(t[0]), float(t[1]), float(t[2])))


def Rt_from_pose3(p: "Pose3") -> Tuple[np.ndarray, np.ndarray]:
    if not GTSAM_OK:
        raise RuntimeError("Rt_from_pose3 requires GTSAM")
    rot = p.rotation()
    if hasattr(rot, "matrix"):
        R = np.array(rot.matrix(), dtype=np.float64)
    else:
        R = np.array(rot, dtype=np.float64).reshape(3, 3)

    trans = p.translation()
    if hasattr(trans, "x"):
        t = np.array([trans.x(), trans.y(), trans.z()], dtype=np.float64)
    else:
        t = np.array(trans, dtype=np.float64).reshape(3)
    return R, t


@dataclass
class Keyframe:
    idx: int
    pose_est: "Pose3"
    kps: List[cv2.KeyPoint]
    desc: np.ndarray
    uv: np.ndarray
    xyz: np.ndarray


class PoseGraphISAM:
    def __init__(self, prior_sigmas: np.ndarray, odom_sigmas: np.ndarray, loop_sigmas: np.ndarray):
        if not GTSAM_OK:
            raise RuntimeError("PoseGraphISAM requires GTSAM")
        self.isam = ISAM2()
        self.graph = NonlinearFactorGraph()
        self.init = Values()
        self.have0 = False
        self.prev_key = None
        self.prev_pose_est = None
        self.NOISE_PRIOR = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
        self.NOISE_ODOM = gtsam.noiseModel.Diagonal.Sigmas(odom_sigmas)
        self.NOISE_LOOP = gtsam.noiseModel.Diagonal.Sigmas(loop_sigmas)

    def add_keyframe(self, k_idx: int, pose_est: "Pose3") -> "Values":
        key = X(k_idx)
        if not self.have0:
            self.graph.add(gtsam.PriorFactorPose3(key, pose_est, self.NOISE_PRIOR))
            self.have0 = True
        elif self.prev_key is not None and self.prev_pose_est is not None:
            odom_meas = self.prev_pose_est.between(pose_est)
            self.graph.add(gtsam.BetweenFactorPose3(self.prev_key, key, odom_meas, self.NOISE_ODOM))
        self.init.insert(key, pose_est)
        self.isam.update(self.graph, self.init)
        self.graph = NonlinearFactorGraph()
        self.init = Values()
        self.prev_key, self.prev_pose_est = key, pose_est
        return self.isam.calculateEstimate()

    def add_loop(self, j_idx: int, i_idx: int, j_T_i: "Pose3") -> "Values":
        key_j = X(j_idx)
        key_i = X(i_idx)
        self.graph.add(gtsam.BetweenFactorPose3(key_j, key_i, j_T_i, self.NOISE_LOOP))
        self.isam.update(self.graph, Values())
        self.graph = NonlinearFactorGraph()
        return self.isam.calculateEstimate()


def extract_orb_with_3d(gray: np.ndarray, points_3d: np.ndarray, valid_mask: np.ndarray, nfeatures: int = 1500):
    orb = cv2.ORB_create(nfeatures=int(nfeatures))
    kps, desc = orb.detectAndCompute(gray, None)
    if desc is None or len(kps) == 0:
        return [], None, None, None
    uv = np.array([kp.pt for kp in kps], dtype=np.float32)
    h, w = gray.shape
    xyz = np.full((len(kps), 3), np.nan, dtype=np.float32)
    for i, (u, v) in enumerate(uv):
        uu, vv = int(round(u)), int(round(v))
        if 0 <= uu < w and 0 <= vv < h and valid_mask[vv, uu]:
            xyz[i] = points_3d[vv, uu]
    return kps, desc, uv, xyz


def match_and_pnp(
    ref_kf: Keyframe,
    cur_kf: Keyframe,
    K: np.ndarray,
    ratio: float = 0.75,
    ransac_px: float = 3.0,
    min_inliers: int = 50,
):
    if ref_kf.desc is None or cur_kf.desc is None:
        return False, None, None, 0, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(cur_kf.desc, ref_kf.desc, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < min_inliers:
        return False, None, None, len(good), None

    obj_pts = []
    img_pts = []
    for m in good:
        X_ref = ref_kf.xyz[m.trainIdx]
        if np.isfinite(X_ref).all():
            obj_pts.append(X_ref)
            img_pts.append(cur_kf.uv[m.queryIdx])

    if len(obj_pts) < min_inliers:
        return False, None, None, len(obj_pts), None

    obj = np.ascontiguousarray(np.array(obj_pts, dtype=np.float32).reshape(-1, 3))
    img = np.ascontiguousarray(np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2))

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj,
        img,
        K,
        None,
        iterationsCount=200,
        reprojectionError=float(ransac_px),
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None or len(inliers) < min_inliers:
        return False, None, None, 0, None

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    return True, R.astype(np.float64), t.astype(np.float64), len(inliers), inliers

# ---------------------------- Commands ----------------------------

def cmd_make_board(args):
    """Generate a printable chessboard as a PDF or PNG. The board will have (cols+1) x (rows+1) squares,
    producing (cols x rows) inner corners used during calibration.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as e:
        print("matplotlib is required for 'make-board'. Install it or create a board manually.\n", e)
        sys.exit(2)

    cols, rows = args.cols, args.rows
    square_mm = args.square_mm
    margin_mm = args.margin_mm
    invert = args.invert

    # Total squares along width/height are +1 of inner corners
    squares_x = cols + 1
    squares_y = rows + 1

    W_mm = squares_x * square_mm + 2*margin_mm
    H_mm = squares_y * square_mm + 2*margin_mm

    fig = plt.figure(figsize=(W_mm/25.4, H_mm/25.4), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_mm)
    ax.set_ylim(0, H_mm)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw background (white by default)
    bg_color = 'white' if not invert else 'black'
    fg_color = 'black' if not invert else 'white'
    ax.add_patch(Rectangle((0, 0), W_mm, H_mm, facecolor=bg_color, edgecolor=bg_color))

    # Draw squares
    origin_x = margin_mm
    origin_y = margin_mm
    for y in range(squares_y):
        for x in range(squares_x):
            # Checker pattern
            if (x + y) % 2 == 0:
                color = fg_color
            else:
                color = bg_color
            ax.add_patch(Rectangle((origin_x + x*square_mm, origin_y + y*square_mm),
                                   square_mm, square_mm, facecolor=color, edgecolor=color))

    out = args.out
    root, ext = os.path.splitext(out)
    if ext.lower() not in ['.pdf', '.png']:
        out_pdf = root + '.pdf'
        out_png = root + '.png'
        fig.savefig(out_pdf, dpi=300, transparent=False)
        fig.savefig(out_png, dpi=300, transparent=False)
        print(f"Saved {out_pdf} and {out_png}")
    else:
        fig.savefig(out, dpi=300, transparent=False)
        print(f"Saved {out}")

def cmd_capture(args):
    pat = PatternSpec(cols=args.cols, rows=args.rows, square_size=float(args.square))
    pattern_size = (pat.cols, pat.rows)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == 'nt' else 0)
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print("Failed to open camera index", args.camera)
        sys.exit(2)

    objpoints = []   # list of (N,3)
    imgpoints_l = [] # list of (N,1,2)
    imgpoints_r = []
    n_saved = 0
    last_save_time = 0.0

    print("\nCAPTURE MODE")
    print(" - Move the printed board around (distance, tilt, rotation) to cover the full FOV of both eyes.")
    print(" - Press SPACE to save a sample when corners are found in BOTH views.")
    print(" - Press 'u' to undo last saved sample.")
    print(" - Press 'q' to finish and write", args.out)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        left, right = split_left_right(frame, mode=args.split, split_px=args.split_px, ratio=args.split_ratio, swap=args.swap)

        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = find_corners(gray_l, pattern_size)
        found_r, corners_r = find_corners(gray_r, pattern_size)

        vis = np.hstack([left.copy(), right.copy()])
        mid = left.shape[1]
        if found_l:
            cv2.drawChessboardCorners(vis[:, :mid], pattern_size, corners_l, True)
        if found_r:
            cv2.drawChessboardCorners(vis[:, mid:], pattern_size, corners_r, True)

        draw_info(vis, [
            f"samples: {n_saved}",
            f"found L: {found_l} | R: {found_r}",
            "SPACE=save  u=undo  q=quit",
        ])

        cv2.imshow("stereo capture (left | right)", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('u'):
            if n_saved > 0:
                objpoints.pop(); imgpoints_l.pop(); imgpoints_r.pop()
                n_saved -= 1
                print("Undid last sample. Now:", n_saved)
        elif key == 32:  # SPACE
            if found_l and found_r:
                objpoints.append(make_object_points(pat))
                imgpoints_l.append(corners_l)
                imgpoints_r.append(corners_r)
                n_saved += 1
                last_save_time = time.time()
                print(f"Saved sample #{n_saved}")
            else:
                print("Corners not found on both sides; not saved.")

    cap.release()
    cv2.destroyAllWindows()

    if n_saved == 0:
        print("No samples captured; nothing to save.")
        return

    # Save dataset
    left_h, left_w = left.shape[:2]
    data = {
        'pattern_cols': pat.cols,
        'pattern_rows': pat.rows,
        'square_size_m': pat.square_size,
        'image_size': (left_w, left_h),  # per-eye size
        'objpoints': objpoints,
        'imgpoints_l': imgpoints_l,
        'imgpoints_r': imgpoints_r,
        'split_mode': args.split,
        'split_px': args.split_px,
        'split_ratio': args.split_ratio,
        'swap_lr': args.swap,
    }

    # Use numpy savez for portability
    np.savez_compressed(args.out, **{k: np.array(v, dtype=object) for k, v in data.items()})
    print(f"Wrote dataset to {args.out} with {n_saved} samples.")

def cmd_calibrate(args):
    # Load dataset
    z = np.load(args.data, allow_pickle=True)
    cols = int(z['pattern_cols'])
    rows = int(z['pattern_rows'])
    square_size = float(z['square_size_m'])
    # Convert stored object arrays back to contiguous float32 buffers for OpenCV
    image_size = tuple(int(v) for v in z['image_size'])
    objpoints = [np.ascontiguousarray(np.asarray(op, dtype=np.float32)) for op in z['objpoints']]
    imgpoints_l = [np.ascontiguousarray(np.asarray(ip, dtype=np.float32)) for ip in z['imgpoints_l']]
    imgpoints_r = [np.ascontiguousarray(np.asarray(ip, dtype=np.float32)) for ip in z['imgpoints_r']]

    if len(objpoints) == 0:
        print("Dataset is empty.")
        sys.exit(2)

    print(f"Loaded {len(objpoints)} stereo samples; per-eye image size = {image_size}")

    # Calibrate intrinsics separately
    flags = 0
    if args.fix_principal_point: flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    if args.zero_tangent_dist:   flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if args.fix_k3:              flags |= cv2.CALIB_FIX_K3
    if args.rational_model:      flags |= cv2.CALIB_RATIONAL_MODEL

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)

    print("\nCalibrating LEFT intrinsics...")
    ret_l, K1, D1, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, image_size, None, None,
                                                          flags=flags, criteria=criteria)
    rms_l, per_view_l = compute_reprojection_error(objpoints, imgpoints_l, K1, D1, rvecs_l, tvecs_l)
    print(f"LEFT RMS reprojection error: {rms_l:.4f} px")

    print("\nCalibrating RIGHT intrinsics...")
    ret_r, K2, D2, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, image_size, None, None,
                                                          flags=flags, criteria=criteria)
    rms_r, per_view_r = compute_reprojection_error(objpoints, imgpoints_r, K2, D2, rvecs_r, tvecs_r)
    print(f"RIGHT RMS reprojection error: {rms_r:.4f} px")

    # filter out outlier calibration frames
    max_view_err = 0.25  # px, tune

    good_idx = [
        i for i, (el, er) in enumerate(zip(per_view_l, per_view_r))
        if el < max_view_err and er < max_view_err
    ]

    objpoints   = [objpoints[i]   for i in good_idx]
    imgpoints_l = [imgpoints_l[i] for i in good_idx]
    imgpoints_r = [imgpoints_r[i] for i in good_idx]

    print(f"Filtered to {len(objpoints)} good views (<= {max_view_err} px).")

    # Stereo calibration (extrinsics)
    print("\nStereo calibration (estimating R, T, E, F)...")
    stereo_flags = cv2.CALIB_FIX_INTRINSIC if args.fix_intrinsic else 0
    ret_stereo, K1s, D1s, K2s, D2s, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, K1, D1, K2, D2, image_size,
        flags=stereo_flags, criteria=criteria
    )
    print(f"Stereo RMS reprojection error: {ret_stereo:.4f} px")
    baseline = np.linalg.norm(T)
    print(f"Baseline ||T|| = {baseline:.6f} meters")
    print("R =\n", R)
    print("T =\n", T.reshape(-1))

    # Rectification
    alpha = args.alpha  # 0 => crop max; 1 => keep all; -1 => default
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1s, D1s, K2s, D2s, image_size, R, T, alpha=alpha)
    print("\nRectification computed. Q matrix:\n", Q)

    # Save everything
    out = {
        'K1': K1s, 'D1': D1s, 'K2': K2s, 'D2': D2s,
        'R': R, 'T': T, 'E': E, 'F': F,
        'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
        'image_size': image_size,
        'per_view_error_left': np.array(per_view_l),
        'per_view_error_right': np.array(per_view_r),
        'rms_left': float(rms_l),
        'rms_right': float(rms_r),
        'rms_stereo': float(ret_stereo),
    }
    np.savez_compressed(args.out, **out)
    print(f"\nWrote calibration to {args.out}")

def cmd_preview(args):
    # Load calibration and optionally live-preview rectification
    calib = np.load(args.calib, allow_pickle=True)
    K1, D1 = calib['K1'], calib['D1']
    K2, D2 = calib['K2'], calib['D2']
    R1, R2, P1, P2 = calib['R1'], calib['R2'], calib['P1'], calib['P2']
    image_size = tuple(int(v) for v in calib['image_size'])

    baseline = float(np.linalg.norm(calib['T'])) if 'T' in calib else None
    print("\nCalibration summary:")
    print(f"  image_size: {image_size}")
    print(f"  K1 fx, fy: {K1[0,0]:.3f}, {K1[1,1]:.3f} | cx, cy: {K1[0,2]:.3f}, {K1[1,2]:.3f}")
    print(f"  K2 fx, fy: {K2[0,0]:.3f}, {K2[1,1]:.3f} | cx, cy: {K2[0,2]:.3f}, {K2[1,2]:.3f}")
    if baseline is not None:
        print(f"  baseline |T|: {baseline:.6f} m")
    if 'rms_left' in calib and 'rms_right' in calib and 'rms_stereo' in calib:
        print(f"  rms (left/right/stereo): {calib['rms_left']:.4f} / {calib['rms_right']:.4f} / {calib['rms_stereo']:.4f} px")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == 'nt' else 0)
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print("Failed to open camera", args.camera)
        sys.exit(2)

    print("Press 'q' to quit preview.")

    map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)

    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    last_report_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break
        left, right = split_left_right(frame, mode=args.split, split_px=args.split_px, ratio=args.split_ratio, swap=args.swap)
        left_r = cv2.remap(left, map1_l, map2_l, cv2.INTER_LINEAR)
        right_r = cv2.remap(right, map1_r, map2_r, cv2.INTER_LINEAR)

        gray_l = left_r if left_r.ndim == 2 else cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY)
        gray_r = right_r if right_r.ndim == 2 else cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY)
        kp_l, desc_l = orb.detectAndCompute(gray_l, None)
        kp_r, desc_r = orb.detectAndCompute(gray_r, None)

        vertical_err_median = None
        vertical_err_std = None
        matches_used = 0

        if desc_l is not None and desc_r is not None and len(kp_l) > 0 and len(kp_r) > 0:
            try:
                matches = bf.match(desc_l, desc_r)
            except cv2.error:
                matches = []
            if matches:
                matches = sorted(matches, key=lambda m: m.distance)
                best_matches = matches[:200]
                if best_matches:
                    y_errors = np.array(
                        [abs(kp_l[m.queryIdx].pt[1] - kp_r[m.trainIdx].pt[1]) for m in best_matches],
                        dtype=np.float32,
                    )
                    if y_errors.size:
                        vertical_err_median = float(np.median(y_errors))
                        vertical_err_std = float(np.std(y_errors))
                        matches_used = len(best_matches)
                        now = time.time()
                        if now - last_report_time >= 1.0:
                            print(
                                f"Vertical rectification error: median={vertical_err_median:.3f}px "
                                f"stddev={vertical_err_std:.3f}px from {matches_used} ORB matches"
                            )
                            last_report_time = now

        vis = np.hstack([left_r, right_r])
        # vis = np.hstack([left, right])

        # draw some horizontal lines to check epipolar alignment
        h = vis.shape[0]
        for y in np.linspace(20, h-20, 10).astype(int):
            cv2.line(vis, (0, y), (vis.shape[1]-1, y), (0,255,0), 1, cv2.LINE_AA)

        info_lines = []
        if vertical_err_median is not None and vertical_err_std is not None:
            info_lines.append(f"ORB matches: {matches_used}")
            info_lines.append(f"Rect err median: {vertical_err_median:.2f}px")
            info_lines.append(f"Rect err stddev: {vertical_err_std:.2f}px")
        else:
            info_lines.append(f"ORB matches: {matches_used}")
            info_lines.append("Rect err median: n/a")
            info_lines.append("Rect err stddev: n/a")
        draw_info(vis, info_lines)

        cv2.imshow("rectified preview (left | right)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def cmd_preview_depth(args):
    rr = None
    rerun_import_error = None
    if getattr(args, "log_rerun", True):
        try:
            import rerun as rr  # type: ignore
        except Exception as exc:
            rerun_import_error = exc

    calib = np.load(args.calib, allow_pickle=True)
    required = ['K1', 'D1', 'K2', 'D2', 'R1', 'R2', 'P1', 'P2', 'Q', 'image_size']
    missing = [k for k in required if k not in calib]
    if missing:
        print(f"Calibration file is missing keys required for depth preview: {missing}")
        sys.exit(2)

    K1, D1 = calib['K1'], calib['D1']
    K2, D2 = calib['K2'], calib['D2']
    R1, R2, P1, P2 = calib['R1'], calib['R2'], calib['P1'], calib['P2']
    Q = calib['Q']
    image_size = tuple(int(v) for v in calib['image_size'])
    cam_matrix = P1[:3, :3]

    baseline = float(np.linalg.norm(calib['T'])) if 'T' in calib else None

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == 'nt' else 0)
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print("Failed to open camera", args.camera)
        sys.exit(2)

    rerun_enabled = rr is not None and getattr(args, "log_rerun", True)
    if rerun_enabled:
        rr.init(args.rr_app_id, spawn=args.spawn_viewer)
        rr.log(args.rr_space, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(
            f"{args.rr_space}/{args.rr_camera_path}",
            rr.Pinhole(
                focal_length=[float(cam_matrix[0, 0]), float(cam_matrix[1, 1])],
                principal_point=[float(cam_matrix[0, 2]), float(cam_matrix[1, 2])],
                width=image_size[0],
                height=image_size[1],
            ),
            static=True,
        )
    elif rerun_import_error and getattr(args, "log_rerun", True):
        print(f"rerun-sdk is unavailable; skipping rerun logging ({rerun_import_error}).")

    print("Press 'q' to quit depth preview.")

    map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)

    depth_engine = getattr(args, "depth_engine", "stereo")
    use_metric3d = depth_engine == "metric3d"
    metric3d_runner = None
    metric_conf_thresh = float(getattr(args, "metric3d_min_confidence", 0.0)) if use_metric3d else 0.0
    metric_info_line = None

    stereo = None
    last_sgbm_params = None
    control_win = None

    if use_metric3d:
        metric_model = getattr(args, "metric3d_model", "metric3d_vit_small")
        metric_device = getattr(args, "metric3d_device", None)
        try:
            metric3d_runner = get_metric3d_runner(metric_model, metric_device)
        except RuntimeError as exc:
            print(exc)
            sys.exit(2)
        metric_device_str = str(metric3d_runner.device)
        metric_info_line = f"Metric3D model: {metric3d_runner.model_name} ({metric_device_str})"
        print(f"Using Metric3D depth backend ({metric_info_line})")
    else:
        num_disp = max(16, int(math.ceil(args.num_disparities / 16) * 16))
        block_size = int(args.block_size)
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, min(block_size, 11))

        def build_sgbm(min_disp: int, num_disp_val: int, block_val: int, uniq_val: int, speckle_window_val: int, speckle_range_val: int, pre_filter_val: int):
            return cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=num_disp_val,
                blockSize=block_val,
                P1=8 * 3 * block_val * block_val,
                P2=32 * 3 * block_val * block_val,
                disp12MaxDiff=args.disp12_max_diff,
                preFilterCap=pre_filter_val,
                uniquenessRatio=uniq_val,
                speckleWindowSize=speckle_window_val,
                speckleRange=speckle_range_val,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )

        control_win = "depth controls"
        cv2.namedWindow(control_win, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(control_win, 400, 260)
        except Exception:
            pass

        def _noop(_: int) -> None:
            return

        MIN_DISP_OFFSET = 256
        NUM_DISP_MAX_UNITS = 32
        BLOCK_SIZE_MAX_INDEX = 4
        SPECKLE_WINDOW_MAX = 400
        SPECKLE_RANGE_MAX = 10
        PREFILTER_CAP_MAX = 63
        INVALID_DISP_MAX = 200
        MAX_DEPTH_SLIDER_MAX = 300

        def clamp_to_slider(val: int, max_val: int) -> int:
            return max(0, min(int(val), max_val))

        min_disp_slider = clamp_to_slider(args.min_disparity + MIN_DISP_OFFSET, MIN_DISP_OFFSET * 2)
        cv2.createTrackbar("min_disp", control_win, min_disp_slider, MIN_DISP_OFFSET * 2, _noop)

        initial_units = max(1, num_disp // 16)
        cv2.createTrackbar("num_disp_x16", control_win, clamp_to_slider(initial_units, NUM_DISP_MAX_UNITS), NUM_DISP_MAX_UNITS, _noop)

        initial_block_idx = clamp_to_slider((block_size - 3) // 2, BLOCK_SIZE_MAX_INDEX)
        cv2.createTrackbar("block_size", control_win, initial_block_idx, BLOCK_SIZE_MAX_INDEX, _noop)

        cv2.createTrackbar("uniq_ratio", control_win, clamp_to_slider(args.uniqueness_ratio, 50), 50, _noop)
        cv2.createTrackbar("speckle_window", control_win, clamp_to_slider(args.speckle_window, SPECKLE_WINDOW_MAX), SPECKLE_WINDOW_MAX, _noop)
        cv2.createTrackbar("speckle_range", control_win, clamp_to_slider(args.speckle_range, SPECKLE_RANGE_MAX), SPECKLE_RANGE_MAX, _noop)

        pre_filter_init = max(1, min(int(args.pre_filter_cap), PREFILTER_CAP_MAX))
        cv2.createTrackbar("pre_filter_cap", control_win, pre_filter_init, PREFILTER_CAP_MAX, _noop)

        invalid_disp_init = clamp_to_slider(int(round(max(args.invalid_disparity, 0.0) * 10.0)), INVALID_DISP_MAX)
        cv2.createTrackbar("invalid_disp_x10", control_win, invalid_disp_init, INVALID_DISP_MAX, _noop)

        max_depth_init = clamp_to_slider(int(round(max(args.max_depth, 0.1) * 10.0)), MAX_DEPTH_SLIDER_MAX)
        cv2.createTrackbar("max_depth_x10", control_win, max_depth_init, MAX_DEPTH_SLIDER_MAX, _noop)

        def read_controls():
            min_disp_val = cv2.getTrackbarPos("min_disp", control_win) - MIN_DISP_OFFSET
            num_disp_units = max(1, cv2.getTrackbarPos("num_disp_x16", control_win))
            num_disp_val = num_disp_units * 16
            block_idx = cv2.getTrackbarPos("block_size", control_win)
            block_val = 3 + 2 * max(0, min(block_idx, BLOCK_SIZE_MAX_INDEX))
            uniq_val = cv2.getTrackbarPos("uniq_ratio", control_win)
            speckle_window_val = cv2.getTrackbarPos("speckle_window", control_win)
            speckle_range_val = cv2.getTrackbarPos("speckle_range", control_win)
            pre_filter_val = max(1, cv2.getTrackbarPos("pre_filter_cap", control_win))
            invalid_disp_val = cv2.getTrackbarPos("invalid_disp_x10", control_win) / 10.0
            max_depth_val = cv2.getTrackbarPos("max_depth_x10", control_win) / 10.0
            max_depth_val = max(0.1, max_depth_val)
            return {
                'min_disp': int(min_disp_val),
                'num_disp': int(num_disp_val),
                'block_size': int(block_val),
                'uniqueness': int(uniq_val),
                'speckle_window': int(speckle_window_val),
                'speckle_range': int(speckle_range_val),
                'pre_filter_cap': int(pre_filter_val),
                'invalid_disp': float(invalid_disp_val),
                'max_depth': float(max_depth_val),
            }

        ctrl = read_controls()
        stereo = build_sgbm(
            ctrl['min_disp'],
            ctrl['num_disp'],
            ctrl['block_size'],
            ctrl['uniqueness'],
            ctrl['speckle_window'],
            ctrl['speckle_range'],
            ctrl['pre_filter_cap'],
        )
        last_sgbm_params = (
            ctrl['min_disp'],
            ctrl['num_disp'],
            ctrl['block_size'],
            ctrl['uniqueness'],
            ctrl['speckle_window'],
            ctrl['speckle_range'],
            ctrl['pre_filter_cap'],
        )

    fps_ts = time.time()
    frame_counter = 0
    fps_display = 0.0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break
        frame_idx += 1

        left, right = split_left_right(frame, mode=args.split, split_px=args.split_px, ratio=args.split_ratio, swap=args.swap)
        left_r = cv2.remap(left, map1_l, map2_l, cv2.INTER_LINEAR)
        right_r = cv2.remap(right, map1_r, map2_r, cv2.INTER_LINEAR)

        disparity = None
        conf_map = None

        if use_metric3d:
            depth_map, conf_map = metric3d_runner.infer_depth(left_r, cam_matrix)
            depth_map = depth_map.astype(np.float32, copy=False)
            valid_mask = np.isfinite(depth_map) & (depth_map > 0.0)
            if conf_map is not None and metric_conf_thresh > 0.0:
                valid_mask &= conf_map >= metric_conf_thresh
            depth_map[~valid_mask] = 0.0
            vis_depth_limit = float(args.max_depth)
        else:
            gray_l = cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY)

            ctrl = read_controls()
            current_sgbm_params = (
                ctrl['min_disp'],
                ctrl['num_disp'],
                ctrl['block_size'],
                ctrl['uniqueness'],
                ctrl['speckle_window'],
                ctrl['speckle_range'],
                ctrl['pre_filter_cap'],
            )
            if current_sgbm_params != last_sgbm_params:
                stereo = build_sgbm(
                    ctrl['min_disp'],
                    ctrl['num_disp'],
                    ctrl['block_size'],
                    ctrl['uniqueness'],
                    ctrl['speckle_window'],
                    ctrl['speckle_range'],
                    ctrl['pre_filter_cap'],
                )
                last_sgbm_params = current_sgbm_params

            disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0
            valid_mask = disparity > ctrl['invalid_disp']
            points_3d = cv2.reprojectImageTo3D(disparity, Q)
            depth_map = points_3d[:, :, 2].astype(np.float32, copy=False)
            valid_mask &= np.isfinite(depth_map) & (depth_map > 0.0)
            depth_map[~valid_mask] = 0.0
            vis_depth_limit = float(ctrl['max_depth'])

        has_valid = bool(valid_mask.any())
        if not math.isfinite(vis_depth_limit) or vis_depth_limit <= 0.0:
            vis_depth_limit = float(np.nanmax(depth_map[valid_mask])) if has_valid else 1.0
        vis_depth_limit = max(1e-6, vis_depth_limit)

        depth_clipped = np.clip(depth_map, 0.0, vis_depth_limit)
        depth_norm = np.nan_to_num(depth_clipped / vis_depth_limit)
        depth_vis = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        depth_color[~valid_mask] = (0, 0, 0)

        if rerun_enabled:
            depth_for_log = np.where(valid_mask, depth_map, 0.0).astype(np.float32)
            rr.set_time("frame", sequence=frame_idx)
            rr.log(
                f"{args.rr_space}/{args.rr_camera_path}/depth",
                rr.DepthImage(depth_for_log, meter=1.0),
            )

        frame_counter += 1
        if frame_counter >= 15:
            now = time.time()
            fps_display = frame_counter / (now - fps_ts)
            fps_ts = now
            frame_counter = 0

        info_lines: List[str] = []
        if use_metric3d:
            depth_valid = depth_map[valid_mask]
            depth_min = float(depth_valid.min()) if depth_valid.size else 0.0
            depth_max = float(depth_valid.max()) if depth_valid.size else 0.0
            info_lines.append(f"Metric3D depth min/max: {depth_min:.2f} / {depth_max:.2f} m")
            info_lines.append(f"vis range: 0 - {vis_depth_limit:.2f} m")
            if metric_info_line:
                info_lines.append(metric_info_line)
            if metric_conf_thresh > 0.0 and conf_map is not None:
                info_lines.append(f"confidence >= {metric_conf_thresh:.2f}")
        else:
            disp_valid = disparity[valid_mask] if disparity is not None and has_valid else None
            disp_min = float(np.nanmin(disp_valid)) if disp_valid is not None and disp_valid.size else 0.0
            disp_max = float(np.nanmax(disp_valid)) if disp_valid is not None and disp_valid.size else 0.0
            info_lines.append(f"disparity min/max: {disp_min:.2f} / {disp_max:.2f}")
            info_lines.append(f"depth range: 0 - {vis_depth_limit:.2f} m")
            info_lines.append(f"SGBM min/num/block: {ctrl['min_disp']} / {ctrl['num_disp']} / {ctrl['block_size']}")
            info_lines.append(f"uniq={ctrl['uniqueness']} speckle={ctrl['speckle_window']}/{ctrl['speckle_range']} inv<={ctrl['invalid_disp']:.1f}")

        if baseline is not None:
            info_lines.append(f"baseline: {baseline:.4f} m")
        if fps_display:
            info_lines.append(f"fps: {fps_display:.1f}")

        depth_display = depth_color.copy()
        draw_info(depth_display, info_lines)

        vis = np.hstack([left_r, depth_display])
        cv2.imshow("rectified (left) | depth", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def cmd_preview_odometry(args):
    try:
        import rerun as rr
    except Exception as exc:
        print("rerun-sdk is required for preview-odometry:", exc)
        sys.exit(2)

    calib = np.load(args.calib, allow_pickle=True)
    required = ['K1', 'D1', 'K2', 'D2', 'R1', 'R2', 'P1', 'P2', 'Q', 'image_size']
    missing = [k for k in required if k not in calib]
    if missing:
        print(f"Calibration file is missing keys required for odometry preview: {missing}")
        sys.exit(2)

    K1, D1 = calib['K1'], calib['D1']
    K2, D2 = calib['K2'], calib['D2']
    R1, R2, P1, P2 = calib['R1'], calib['R2'], calib['P1'], calib['P2']
    Q = calib['Q']
    image_size = tuple(int(v) for v in calib['image_size'])

    cam_matrix = P1[:3, :3].astype(np.float32)

    depth_engine = getattr(args, "depth_engine", "stereo")
    use_metric3d = depth_engine == "metric3d"
    metric3d_runner = None
    metric_conf_thresh = float(getattr(args, "metric3d_min_confidence", 0.0)) if use_metric3d else 0.0
    metric_info_line = None
    metric_rays: Optional[Tuple[np.ndarray, np.ndarray]] = None

    if use_metric3d:
        metric_model = getattr(args, "metric3d_model", "metric3d_vit_small")
        metric_device = getattr(args, "metric3d_device", None)
        try:
            metric3d_runner = get_metric3d_runner(metric_model, metric_device)
        except RuntimeError as exc:
            print(exc)
            sys.exit(2)
        metric_device_str = str(metric3d_runner.device)
        metric_info_line = f"Metric3D model: {metric3d_runner.model_name} ({metric_device_str})"
        print(f"Using Metric3D depth backend ({metric_info_line})")
        metric_rays = compute_pixel_rays((image_size[1], image_size[0]), cam_matrix)

    use_lc = bool(getattr(args, "use_loop_closure", True)) and GTSAM_OK
    kf_nfeatures = int(getattr(args, "kf_nfeatures", 1500))
    kf_min_interval = int(getattr(args, "kf_min_interval", 12))
    kf_trans_thresh = float(getattr(args, "kf_trans_thresh", 0.20))
    kf_rot_thresh_deg = float(getattr(args, "kf_rot_thresh_deg", 10.0))
    lc_exclusion = int(getattr(args, "lc_exclusion", 30))
    lc_ratio = float(getattr(args, "lc_match_ratio", 0.75))
    lc_ransac_px = float(getattr(args, "lc_ransac_px", 2.0))
    lc_min_inliers = int(getattr(args, "lc_min_inliers", 60))

    odom_rot_sigma = math.radians(float(getattr(args, "opt_odo_rot_sigma_deg", 2.0)))
    odom_trans_sigma = float(getattr(args, "opt_odo_trans_sigma", 0.05))
    loop_rot_sigma = math.radians(float(getattr(args, "opt_loop_rot_sigma_deg", 1.0)))
    loop_trans_sigma = float(getattr(args, "opt_loop_trans_sigma", 0.02))
    prior_sigmas = np.array([math.radians(30)] * 3 + [5, 5, 5], dtype=np.float64)
    odom_sigmas = np.array([odom_rot_sigma] * 3 + [odom_trans_sigma] * 3, dtype=np.float64)
    loop_sigmas = np.array([loop_rot_sigma] * 3 + [loop_trans_sigma] * 3, dtype=np.float64)

    T_corr = np.eye(4, dtype=np.float64)
    keyframes: List[Keyframe] = []
    loop_edges = []
    pg = PoseGraphISAM(prior_sigmas, odom_sigmas, loop_sigmas) if use_lc else None
    last_kf_idx = -1
    last_kf_frame = -999
    last_kf_pose_est = None

    def should_make_kf(last_pose_R, last_pose_t, curr_R, curr_t, last_frame_idx, this_frame_idx):
        if this_frame_idx - last_frame_idx < kf_min_interval:
            return False
        if last_pose_R is None or last_pose_t is None:
            return True
        dtrans = np.linalg.norm(curr_t - last_pose_t)
        R_rel = last_pose_R.T @ curr_R
        drot = math.degrees(np.arccos(max(-1.0, min(1.0, (np.trace(R_rel) - 1.0) / 2.0))))
        return (dtrans > kf_trans_thresh) or (drot > kf_rot_thresh_deg)

    rr.init(args.rr_app_id, spawn=args.spawn_viewer)
    rr.log(args.rr_space, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        f"{args.rr_space}/{args.rr_camera_path}",
        rr.Pinhole(
            focal_length=[float(cam_matrix[0, 0]), float(cam_matrix[1, 1])],
            principal_point=[float(cam_matrix[0, 2]), float(cam_matrix[1, 2])],
            width=image_size[0],
            height=image_size[1],
        ),
        static=True,
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == 'nt' else 0)
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        print("Failed to open camera", args.camera)
        sys.exit(2)

    map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)

    stereo = None
    if not use_metric3d:
        num_disp = max(16, int(math.ceil(args.num_disparities / 16) * 16))
        block_size = int(args.block_size)
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, min(block_size, 11))

        stereo = cv2.StereoSGBM_create(
            minDisparity=args.min_disparity,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size * block_size,
            P2=32 * 3 * block_size * block_size,
            disp12MaxDiff=args.disp12_max_diff,
            preFilterCap=args.pre_filter_cap,
            uniquenessRatio=args.uniqueness_ratio,
            speckleWindowSize=args.speckle_window,
            speckleRange=args.speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    R_wc = np.eye(3, dtype=np.float64)
    t_wc = np.zeros((3, 1), dtype=np.float64)
    trajectory = [np.zeros(3, dtype=np.float32)]
    R_cw_live = np.eye(3, dtype=np.float64)
    t_cw_live = np.zeros(3, dtype=np.float64)

    rr.set_time("frame", sequence=0)
    rr.log(
        f"{args.rr_space}/{args.rr_camera_path}",
        rr.Transform3D(
            translation=[0.0, 0.0, 0.0],
            rotation=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0]),
            relation=rr.TransformRelation.ParentFromChild,
        ),
    )
    rr.log(
        f"{args.rr_space}/trajectory",
        rr.LineStrips3D([np.asarray(trajectory, dtype=np.float32)]),
    )

    prev_gray = None
    prev_pts = None
    prev_pts3d = None

    frame_ts = time.time()
    frame_counter = 0
    fps_display = 0.0
    frame_idx = 0

    print("Press 'q' to quit odometry preview.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed.")
            break

        frame_idx += 1
        tracks_logged = False

        left, right = split_left_right(frame, mode=args.split, split_px=args.split_px, ratio=args.split_ratio, swap=args.swap)
        left_r = cv2.remap(left, map1_l, map2_l, cv2.INTER_LINEAR)
        right_r = cv2.remap(right, map1_r, map2_r, cv2.INTER_LINEAR)

        gray_l = cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray_l.shape

        disparity = None

        if use_metric3d and metric3d_runner is not None:
            depth_map, conf_map = metric3d_runner.infer_depth(left_r, cam_matrix)
            depth_map = depth_map.astype(np.float32, copy=False)
            valid_mask = np.isfinite(depth_map) & (depth_map > 0.0)
            if conf_map is not None and metric_conf_thresh > 0.0:
                valid_mask &= conf_map >= metric_conf_thresh
            if metric_rays is None:
                metric_rays = compute_pixel_rays((image_size[1], image_size[0]), cam_matrix)
            x_norm, y_norm = metric_rays
            points_3d = depth_to_points_from_rays(depth_map, x_norm, y_norm)
            points_3d[~valid_mask] = 0.0
            depth_map[~valid_mask] = 0.0
            depth_for_log = np.where(valid_mask, depth_map, 0.0).astype(np.float32)
        else:
            disparity = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0  # type: ignore[arg-type]
            valid_disp = disparity > args.invalid_disparity
            points_3d = cv2.reprojectImageTo3D(disparity, Q)
            depth_map = points_3d[:, :, 2].astype(np.float32, copy=False)
            valid_depth = np.isfinite(depth_map) & (depth_map > 0)
            valid_mask = valid_disp & valid_depth
            points_3d[~valid_mask] = 0.0
            depth_map[~valid_mask] = 0.0
            depth_for_log = np.where(valid_mask, depth_map, 0.0).astype(np.float32)

        feature_mask = (valid_mask.astype(np.uint8) * 255)

        rr.set_time("frame", sequence=frame_idx)
        rr.log(
            f"{args.rr_space}/{args.rr_camera_path}/left_rect",
            rr.Image(cv2.cvtColor(left_r, cv2.COLOR_BGR2RGB)),
        )
        rr.log(
            f"{args.rr_space}/{args.rr_camera_path}/depth",
            rr.DepthImage(depth_for_log, meter=1.0),
        )

        need_bootstrap = prev_gray is None

        if not need_bootstrap and prev_pts is not None and len(prev_pts) >= args.min_pnp_points:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray_l,
                prev_pts,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            status = status.reshape(-1) if status is not None else np.array([], dtype=np.uint8)

            matched_prev = prev_pts[status == 1] if status.size else np.empty((0, 1, 2), dtype=np.float32)
            matched_next = next_pts[status == 1] if status.size else np.empty((0, 1, 2), dtype=np.float32)
            matched_prev3d = prev_pts3d[status == 1] if status.size else np.empty((0, 3), dtype=np.float32)

            filtered_prev2d = []
            filtered_curr2d = []
            filtered_prev3d = []
            filtered_curr3d = []

            for idx_pt, (pt_prev, pt_curr) in enumerate(zip(matched_prev.reshape(-1, 2), matched_next.reshape(-1, 2))):
                u = int(round(float(pt_curr[0])))
                v = int(round(float(pt_curr[1])))
                if u < 0 or u >= w_img or v < 0 or v >= h_img:
                    continue
                if not valid_mask[v, u]:
                    continue
                filtered_prev2d.append(pt_prev)
                filtered_curr2d.append(pt_curr)
                filtered_prev3d.append(matched_prev3d[idx_pt])
                filtered_curr3d.append(points_3d[v, u])

            if filtered_prev3d:
                obj_points = np.asarray(filtered_prev3d, dtype=np.float32)
                img_points = np.asarray(filtered_curr2d, dtype=np.float32).reshape(-1, 1, 2)
                ret, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_points,
                    img_points,
                    cam_matrix,
                    None,
                    iterationsCount=args.pnp_iterations,
                    reprojectionError=args.pnp_reproj_error,
                    confidence=0.999,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ret and inliers is not None and len(inliers) >= args.min_pnp_points:
                    inliers_mask = inliers.reshape(-1)
                    curr_inliers = np.asarray(filtered_curr3d, dtype=np.float32)[inliers_mask]

                    R_rel, _ = cv2.Rodrigues(rvec)
                    t_rel = tvec.reshape(3, 1)

                    R_wc = R_rel @ R_wc
                    t_wc = R_rel @ t_wc + t_rel

                    R_cw = R_wc.T
                    t_cw = (-R_cw @ t_wc).reshape(3)
                    T_cw_est = Rt_to_T(R_cw, t_cw)
                    T_cw_live = compose(T_corr, T_cw_est)
                    R_cw_live, t_cw_live = T_to_Rt(T_cw_live)
                    trajectory.append(t_cw.astype(np.float32))

                    quat = rotation_matrix_to_quaternion(R_cw_live)
                    norm_q = np.linalg.norm(quat)
                    quat = quat / norm_q if norm_q > 1e-6 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    rr.log(
                        f"{args.rr_space}/{args.rr_camera_path}",
                        rr.Transform3D(
                            translation=t_cw_live.tolist(),
                            quaternion=quat.tolist(),
                            relation=rr.TransformRelation.ParentFromChild,
                        ),
                    )
                    rr.log(
                        f"{args.rr_space}/{args.rr_camera_path}/tracks",
                        rr.Points3D(curr_inliers.astype(np.float32)),
                    )
                    tracks_logged = True
                    rr.log(
                        f"{args.rr_space}/trajectory",
                        rr.LineStrips3D([np.asarray(trajectory, dtype=np.float32)]),
                    )

                    if use_lc:
                        if last_kf_pose_est is None:
                            last_R = None
                            last_t = None
                        else:
                            last_R, last_t = Rt_from_pose3(last_kf_pose_est)
                        need_kf = (last_kf_idx < 0) or should_make_kf(
                            last_R,
                            last_t,
                            R_cw,
                            t_cw,
                            last_kf_frame,
                            frame_idx,
                        )
                    else:
                        need_kf = False

                    if use_lc and need_kf:
                        kps, desc, uv, xyz = extract_orb_with_3d(
                            gray_l,
                            points_3d,
                            valid_mask,
                            nfeatures=kf_nfeatures,
                        )
                        pose_est = pose3_from_Rt(R_cw, t_cw)
                        kf = Keyframe(
                            idx=last_kf_idx + 1,
                            pose_est=pose_est,
                            kps=kps,
                            desc=desc,
                            uv=uv,
                            xyz=xyz,
                        )
                        keyframes.append(kf)
                        last_kf_idx += 1
                        last_kf_frame = frame_idx
                        last_kf_pose_est = pose_est

                        values = pg.add_keyframe(kf.idx, kf.pose_est)
                        pose_opt = values.atPose3(X(kf.idx))
                        R_opt, t_opt = Rt_from_pose3(pose_opt)
                        T_corr = compose(Rt_to_T(R_opt, t_opt), inv(T_cw_est))
                        T_cw_live = compose(T_corr, T_cw_est)
                        R_cw_live, t_cw_live = T_to_Rt(T_cw_live)
                        quat = rotation_matrix_to_quaternion(R_cw_live)
                        norm_q = np.linalg.norm(quat)
                        quat = quat / norm_q if norm_q > 1e-6 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                        rr.log(
                            f"{args.rr_space}/{args.rr_camera_path}",
                            rr.Transform3D(
                                translation=t_cw_live.tolist(),
                                rotation=rr.Quaternion(xyzw=quat.tolist()),
                                relation=rr.TransformRelation.ParentFromChild,
                            ),
                        )

                        if kf.idx - lc_exclusion >= 0:
                            best = (-1, 0, None, None)
                            for j in range(0, kf.idx - lc_exclusion):
                                ok_lc, R_ji, t_ji, ninl, _ = match_and_pnp(
                                    keyframes[j],
                                    kf,
                                    cam_matrix,
                                    ratio=lc_ratio,
                                    ransac_px=lc_ransac_px,
                                    min_inliers=lc_min_inliers,
                                )
                                if ok_lc and ninl > best[1]:
                                    best = (j, ninl, R_ji, t_ji)

                            j_idx, ninl, R_ji, t_ji = best
                            if j_idx >= 0:
                                j_T_i = pose3_from_Rt(R_ji, t_ji)
                                values = pg.add_loop(j_idx, kf.idx, j_T_i)
                                loop_edges.append((j_idx, kf.idx))

                                pose_opt = values.atPose3(X(kf.idx))
                                R_opt, t_opt = Rt_from_pose3(pose_opt)
                                T_corr = compose(Rt_to_T(R_opt, t_opt), inv(T_cw_est))
                                T_cw_live = compose(T_corr, T_cw_est)
                                R_cw_live, t_cw_live = T_to_Rt(T_cw_live)
                                quat = rotation_matrix_to_quaternion(R_cw_live)
                                norm_q = np.linalg.norm(quat)
                                quat = quat / norm_q if norm_q > 1e-6 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                                rr.log(
                                    f"{args.rr_space}/{args.rr_camera_path}",
                                    rr.Transform3D(
                                        translation=t_cw_live.tolist(),
                                        rotation=rr.Quaternion(xyzw=quat.tolist()),
                                        relation=rr.TransformRelation.ParentFromChild,
                                    ),
                                )

                                opt_pts = []
                                for i in range(kf.idx + 1):
                                    p = values.atPose3(X(i))
                                    _, pt = Rt_from_pose3(p)
                                    opt_pts.append(np.array(pt, dtype=np.float32))
                                if len(opt_pts) >= 2:
                                    rr.log(
                                        f"{args.rr_space}/trajectory_optimized",
                                        rr.LineStrips3D([np.vstack(opt_pts)]),
                                    )

                                for (j0, i0) in loop_edges[-10:]:
                                    pj = np.array(Rt_from_pose3(values.atPose3(X(j0)))[1], dtype=np.float32)
                                    pi = np.array(Rt_from_pose3(values.atPose3(X(i0)))[1], dtype=np.float32)
                                    rr.log(
                                        f"{args.rr_space}/loop_edges/{j0:04d}-{i0:04d}",
                                        rr.LineStrips3D([np.vstack([pj, pi])]),
                                    )

        if not tracks_logged:
            rr.log(
                f"{args.rr_space}/{args.rr_camera_path}/tracks",
                rr.Points3D(np.empty((0, 3), dtype=np.float32)),
            )

        if need_bootstrap or prev_pts is None or len(prev_pts) < args.min_pnp_points:
            prev_pts = None
            prev_pts3d = None

        features = cv2.goodFeaturesToTrack(
            gray_l,
            mask=feature_mask,
            maxCorners=args.max_features,
            qualityLevel=args.feature_quality,
            minDistance=args.feature_min_distance,
            blockSize=7,
        )
        prev_pts, prev_pts3d = gather_depth_samples(features, depth_map, points_3d)
        if len(prev_pts) == 0:
            prev_pts = None
            prev_pts3d = None

        prev_gray = gray_l

        vis = left_r.copy()
        if prev_pts is not None and len(prev_pts) > 0:
            for pt in prev_pts.reshape(-1, 2):
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1, cv2.LINE_AA)

        frame_counter += 1
        if frame_counter >= 20:
            now = time.time()
            fps_display = frame_counter / (now - frame_ts)
            frame_ts = now
            frame_counter = 0

        info = [
            f"tracks: {len(prev_pts) if prev_pts is not None else 0}",
            f"trajectory samples: {len(trajectory)}",
        ]
        latest_pos = t_cw_live.astype(np.float32)
        info.append(f"pos xyz (m): {latest_pos[0]:+.2f} {latest_pos[1]:+.2f} {latest_pos[2]:+.2f}")
        if fps_display:
            info.append(f"fps: {fps_display:.1f}")
        if use_metric3d and metric_info_line:
            info.append(metric_info_line)

        draw_info(vis, info)
        cv2.imshow("stereo odometry (features)", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------- Main ----------------------------

def main():
    p = argparse.ArgumentParser(description="Stereo calibration utility for single-frame (left|right) UVC webcams.")
    sub = p.add_subparsers(dest='cmd', required=True)

    # make-board
    pb = sub.add_parser('make-board', help="Generate a printable chessboard (PDF/PNG).")
    pb.add_argument('--cols', type=int, default=7, help="inner corners along columns (X)")
    pb.add_argument('--rows', type=int, default=10, help="inner corners along rows (Y)")
    pb.add_argument('--square-mm', type=float, default=25.4, help="square size in millimeters")
    pb.add_argument('--margin-mm', type=float, default=10.0, help="margin around the board in millimeters")
    pb.add_argument('--invert', action='store_true', help="invert colors (white squares on black background)")
    pb.add_argument('--out', type=str, default='checkerboard.pdf', help="output file path (.pdf or .png)")
    pb.set_defaults(func=cmd_make_board)

    # capture
    pc = sub.add_parser('capture', help="Capture stereo chessboard samples from the camera.")
    pc.add_argument('--camera', type=int, default=0, help="cv2.VideoCapture index")
    pc.add_argument('--width', type=int, default=1280, help="request frame width (optional)")
    pc.add_argument('--height', type=int, default=640, help="request frame height (optional)")
    pc.add_argument('--fps', type=int, default=0, help="request FPS (optional)")
    pc.add_argument('--cols', type=int, default=7)
    pc.add_argument('--rows', type=int, default=10)
    pc.add_argument('--square', type=float, default=0.0254, help="square size in meters")
    pc.add_argument('--split', type=str, choices=['half','px','ratio'], default='half', help="how to split the wide frame")
    pc.add_argument('--split-px', type=int, default=None, help="column index to split when --split=px")
    pc.add_argument('--split-ratio', type=float, default=0.5, help="ratio (0..1) when --split=ratio")
    pc.add_argument('--swap', action='store_true', help="swap left/right halves")
    pc.add_argument('--out', type=str, default='data.npz', help="output dataset file")
    pc.set_defaults(func=cmd_capture)

    # calibrate
    pal = sub.add_parser('calibrate', help="Calibrate intrinsics/extrinsics from a captured dataset.")
    pal.add_argument('--data', type=str, default='data.npz', help="dataset from 'capture'")
    pal.add_argument('--out', type=str, default='calib.npz', help="output calibration file")
    pal.add_argument('--fix-intrinsic', action='store_true', help="fix intrinsics during stereoCalibrate (recommended)")
    pal.add_argument('--fix-principal-point', action='store_true', help="fix principal point during single-view calibration")
    pal.add_argument('--zero-tangent-dist', action='store_true', help="set tangential distortion p1=p2=0 during single-view calibration")
    pal.add_argument('--fix-k3', action='store_true', help="fix k3=0 during single-view calibration")
    pal.add_argument('--rational-model', action='store_true', help="enable k4..k6 in single-view calibration")
    pal.add_argument('--alpha', type=float, default=0.0, help="rectify free scaling param (-1..1); 0=crop, 1=keep all")
    pal.set_defaults(func=cmd_calibrate)

    # preview
    pp = sub.add_parser('preview', help="Live preview rectified pair using saved calibration.")
    pp.add_argument('--calib', type=str, default='calib.npz', help="calibration file from 'calibrate'")
    pp.add_argument('--camera', type=int, default=0)
    pp.add_argument('--width', type=int, default=1280)
    pp.add_argument('--height', type=int, default=640)
    pp.add_argument('--fps', type=int, default=0)
    pp.add_argument('--split', type=str, choices=['half','px','ratio'], default='half')
    pp.add_argument('--split-px', type=int, default=None)
    pp.add_argument('--split-ratio', type=float, default=0.5)
    pp.add_argument('--swap', action='store_true')
    pp.set_defaults(func=cmd_preview)

    # preview-depth
    ppd = sub.add_parser('preview-depth', aliases=['previewDepth'], help="Preview disparity + depth map using calibration.")
    ppd.add_argument('--calib', type=str, default='calib.npz', help="calibration file from 'calibrate'")
    ppd.add_argument('--camera', type=int, default=0)
    ppd.add_argument('--width', type=int, default=1280)
    ppd.add_argument('--height', type=int, default=640)
    ppd.add_argument('--fps', type=int, default=0)
    ppd.add_argument('--split', type=str, choices=['half','px','ratio'], default='half', help="how to split the wide frame")
    ppd.add_argument('--split-px', type=int, default=None, help="column index to split when --split=px")
    ppd.add_argument('--split-ratio', type=float, default=0.5, help="ratio (0..1) when --split=ratio")
    ppd.add_argument('--swap', action='store_true', help="swap left/right halves")
    ppd.add_argument('--depth-engine', type=str, choices=['stereo', 'metric3d'], default='stereo', help="depth backend: OpenCV StereoSGBM or Metric3Dv2 monocular depth")
    ppd.add_argument('--metric3d-model', type=str, choices=sorted(METRIC3D_MODEL_INPUT_SIZES.keys()), default='metric3d_vit_small', help="Metric3D torch.hub entry name when --depth-engine=metric3d")
    ppd.add_argument('--metric3d-device', type=str, default=None, help="Torch device for Metric3D (auto-detect if omitted)")
    ppd.add_argument('--metric3d-min-confidence', type=float, default=0.0, help="Metric3D confidence threshold (mask depths below this value)")
    ppd.add_argument('--min-disparity', type=int, default=0, help="minimum disparity search range")
    ppd.add_argument('--num-disparities', type=int, default=160, help="number of disparities (multiple of 16)")
    ppd.add_argument('--block-size', type=int, default=7, help="SGBM block size (odd, >=3)")
    ppd.add_argument('--disp12-max-diff', type=int, default=1, help="left-right consistency check threshold")
    ppd.add_argument('--uniqueness-ratio', type=int, default=15, help="uniqueness ratio for disparity")
    ppd.add_argument('--speckle-window', type=int, default=200, help="speckle filter window size")
    ppd.add_argument('--speckle-range', type=int, default=2, help="speckle filter range")
    ppd.add_argument('--pre-filter-cap', type=int, default=45, help="pre-filter cap")
    ppd.add_argument('--invalid-disparity', type=float, default=1.0, help="threshold (<=) to treat disparity as invalid")
    ppd.add_argument('--max-depth', type=float, default=3.0, help="max depth (m) for visualization scale")
    ppd.add_argument('--rr-app-id', type=str, default='stereo_depth_preview', help="rerun application id")
    ppd.add_argument('--rr-space', type=str, default='world', help="rerun space/entity root for logging")
    ppd.add_argument('--rr-camera-path', type=str, default='camera', help="relative path under rr-space for the depth camera entity")
    ppd.add_argument('--log-rerun', action=argparse.BooleanOptionalAction, default=False, help="log rectified depth frames to rerun")
    ppd.add_argument('--spawn-viewer', action=argparse.BooleanOptionalAction, default=True, help="spawn rerun viewer window")
    ppd.set_defaults(func=cmd_preview_depth)

    # preview-odometry
    ppod = sub.add_parser('preview-odometry', aliases=['previewOdometry'], help="Preview stereo odometry streamed to Rerun.")
    ppod.add_argument('--calib', type=str, default='calib.npz', help="calibration file from 'calibrate'")
    ppod.add_argument('--camera', type=int, default=0)
    ppod.add_argument('--width', type=int, default=1280)
    ppod.add_argument('--height', type=int, default=640)
    ppod.add_argument('--fps', type=int, default=0)
    ppod.add_argument('--split', type=str, choices=['half','px','ratio'], default='half', help="how to split the wide frame")
    ppod.add_argument('--split-px', type=int, default=None, help="column index to split when --split=px")
    ppod.add_argument('--split-ratio', type=float, default=0.5, help="ratio (0..1) when --split=ratio")
    ppod.add_argument('--swap', action='store_true', help="swap left/right halves")
    ppod.add_argument('--depth-engine', type=str, choices=['stereo', 'metric3d'], default='stereo', help="depth backend: OpenCV StereoSGBM or Metric3Dv2 monocular depth")
    ppod.add_argument('--metric3d-model', type=str, choices=sorted(METRIC3D_MODEL_INPUT_SIZES.keys()), default='metric3d_vit_small', help="Metric3D torch.hub entry name when --depth-engine=metric3d")
    ppod.add_argument('--metric3d-device', type=str, default=None, help="Torch device for Metric3D (auto-detect if omitted)")
    ppod.add_argument('--metric3d-min-confidence', type=float, default=0.0, help="Metric3D confidence threshold (mask depths below this value)")
    ppod.add_argument('--min-disparity', type=int, default=0, help="minimum disparity search range")
    ppod.add_argument('--num-disparities', type=int, default=128, help="number of disparities (multiple of 16)")
    ppod.add_argument('--block-size', type=int, default=5, help="SGBM block size (odd, >=3)")
    ppod.add_argument('--disp12-max-diff', type=int, default=1, help="left-right consistency check threshold")
    ppod.add_argument('--uniqueness-ratio', type=int, default=10, help="uniqueness ratio for disparity")
    ppod.add_argument('--speckle-window', type=int, default=50, help="speckle filter window size")
    ppod.add_argument('--speckle-range', type=int, default=1, help="speckle filter range")
    ppod.add_argument('--pre-filter-cap', type=int, default=63, help="pre-filter cap")
    ppod.add_argument('--invalid-disparity', type=float, default=0.0, help="threshold (<=) to treat disparity as invalid")
    ppod.add_argument('--max-features', type=int, default=600, help="max corners to track per frame")
    ppod.add_argument('--feature-quality', type=float, default=0.01, help="quality level for feature detection")
    ppod.add_argument('--feature-min-distance', type=float, default=8.0, help="minimum distance between detected features")
    ppod.add_argument('--min-pnp-points', type=int, default=40, help="minimum correspondences before running solvePnP")
    ppod.add_argument('--pnp-iterations', type=int, default=100, help="solvePnP RANSAC iterations")
    ppod.add_argument('--pnp-reproj-error', type=float, default=2.0, help="solvePnP reprojection error threshold")
    ppod.add_argument('--use-loop-closure', action=argparse.BooleanOptionalAction, default=True, help="enable pose graph loop-closure (requires gtsam)")
    ppod.add_argument('--kf-nfeatures', type=int, default=1500, help="ORB keypoints to detect when building keyframes")
    ppod.add_argument('--kf-min-interval', type=int, default=12, help="minimum frames between successive keyframes")
    ppod.add_argument('--kf-trans-thresh', type=float, default=0.20, help="translation threshold (m) to trigger keyframe")
    ppod.add_argument('--kf-rot-thresh-deg', type=float, default=10.0, help="rotation threshold (deg) to trigger keyframe")
    ppod.add_argument('--lc-exclusion', type=int, default=30, help="skip the most recent N keyframes for loop detection")
    ppod.add_argument('--lc-match-ratio', type=float, default=0.75, help="Lowe ratio for ORB loop matching")
    ppod.add_argument('--lc-ransac-px', type=float, default=2.0, help="PnP RANSAC reprojection threshold (px) for loops")
    ppod.add_argument('--lc-min-inliers', type=int, default=60, help="minimum inliers to accept a loop closure")
    ppod.add_argument('--opt-odo-rot-sigma-deg', type=float, default=2.0, help="odometry rotation sigma for pose graph (deg)")
    ppod.add_argument('--opt-odo-trans-sigma', type=float, default=0.05, help="odometry translation sigma for pose graph (m)")
    ppod.add_argument('--opt-loop-rot-sigma-deg', type=float, default=1.0, help="loop closure rotation sigma (deg)")
    ppod.add_argument('--opt-loop-trans-sigma', type=float, default=0.02, help="loop closure translation sigma (m)")
    ppod.add_argument('--rr-app-id', type=str, default='stereo_odometry_preview', help="rerun application id")
    ppod.add_argument('--rr-space', type=str, default='world', help="rerun space/entity root for logging")
    ppod.add_argument('--rr-camera-path', type=str, default='camera', help="relative path under rr-space for the camera entity")
    ppod.add_argument('--spawn-viewer', action=argparse.BooleanOptionalAction, default=True, help="spawn rerun viewer window")
    ppod.set_defaults(func=cmd_preview_odometry)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
