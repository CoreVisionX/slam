"""Thin runner for the refactored stereo SLAM system on KITTI data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import multiprocessing
import sys
import time
from pathlib import Path

import gtsam

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import rerun as rr

from depth.sgbm import SGBM
from registration.registration import FeatureFrame, FramePair, RectifiedStereoFrame, StereoFrame
from registration.utils import solve_pnp
from slam import SlamConfig, create_default_slam_system
from slam.matcher_factory import MatcherType, create_matcher
import tests.test_utils as kitti_utils
from tests.test_utils import (
    get_kitti_calibration,
    get_kitti_iterator_with_odometry,
)


@dataclass(slots=True)
class OdometryEstimate:
    pose: gtsam.Pose3 | None
    match_count: int
    inlier_count: int
    failure_reason: str | None = None


class VisionOdometryEstimator:
    """Estimate frame-to-frame odometry via feature matching + PnP."""

    def __init__(self, matcher_type: MatcherType, *, rectify_inputs: bool) -> None:
        self._matcher = create_matcher(matcher_type)
        self._depth_estimator = SGBM(num_disparities=16 * 4, block_size=5, image_color="RGB")
        self._rectify_inputs = rectify_inputs
        self._prev_feature_frame: FeatureFrame | None = None

    def prime(self, frame: StereoFrame) -> None:
        """Seed the estimator with the very first frame of the sequence."""
        self._prev_feature_frame = self._compute_feature_frame(frame)

    def estimate(self, frame: StereoFrame) -> OdometryEstimate:
        feature_frame = self._compute_feature_frame(frame)
        estimate = OdometryEstimate(pose=None, match_count=0, inlier_count=0)

        if self._prev_feature_frame is not None:
            pair = FramePair(first=self._prev_feature_frame, second=feature_frame)
            matched_pair = self._matcher.match([pair])[0]
            match_count = matched_pair.matches.shape[0]
            pose: gtsam.Pose3 | None = None
            inlier_count = 0
            failure_reason: str | None = None

            if match_count >= 4:
                try:
                    pose, inlier_pair = solve_pnp(matched_pair)
                    inlier_count = inlier_pair.matches.shape[0]
                except Exception as exc:  # noqa: BLE001
                    failure_reason = str(exc)
            else:
                failure_reason = "insufficient_matches"

            estimate = OdometryEstimate(
                pose=pose,
                match_count=match_count,
                inlier_count=inlier_count,
                failure_reason=failure_reason,
            )

        self._restore_feature_storage(feature_frame)
        self._prev_feature_frame = feature_frame
        return estimate

    def _compute_feature_frame(self, frame: StereoFrame) -> FeatureFrame:
        if self._rectify_inputs:
            rectified_frame = frame.rectify()
        elif isinstance(frame, RectifiedStereoFrame):
            rectified_frame = frame
        else:
            rectified_frame = RectifiedStereoFrame(
                left=frame.left,
                right=frame.right,
                left_rect=frame.left,
                right_rect=frame.right,
                calibration=frame.calibration,
            )

        depth_frame = self._depth_estimator.compute_depth(rectified_frame)
        feature_frame = self._matcher.detect_features([depth_frame])[0]
        return feature_frame

    @staticmethod
    def _restore_feature_storage(frame: FeatureFrame | None) -> None:
        if frame is None:
            return

        for key in ("keypoints", "descriptors"):
            value = frame.features.get(key)
            if value is None:
                continue
            if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
                frame.features[key] = value.detach().cpu().numpy()


def _load_first_kitti_frame(sequence_id: str) -> StereoFrame:
    """Load the very first frame so that vision odometry can be primed."""
    sequence = kitti_utils._load_kitti_sequence(sequence_id)  # type: ignore[attr-defined]
    return kitti_utils._load_kitti_frame(sequence, 0)  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the stereo SLAM pipeline on KITTI odometry data.")
    parser.add_argument("--sequence", default="00", help="KITTI odometry sequence id (e.g., 00).")
    parser.add_argument(
        "--keyframe-translation",
        type=float,
        default=1.5,
        help="Translation threshold (meters) for triggering a new keyframe.",
    )
    parser.add_argument(
        "--keyframe-rotation",
        type=float,
        default=24.0,
        help="Rotation threshold (degrees) for triggering a new keyframe.",
    )
    parser.add_argument(
        "--loop-min-inliers",
        type=int,
        default=100,
        help="Minimum inliers required for accepting a loop closure.",
    )
    parser.add_argument(
        "--loop-workers",
        type=int,
        default=None,
        help="Override number of loop closure workers (defaults to heuristic).",
    )
    parser.add_argument(
        "--rerun-tcp",
        type=str,
        default=None,
        help="Optional rerun TCP endpoint (host:port). Leave unset to log locally only.",
    )
    parser.add_argument(
        "--disable-rerun",
        action="store_true",
        help="Disable rerun logging entirely.",
    )
    parser.add_argument(
        "--disable-huber",
        action="store_true",
        help="Disable Huber loss on loop closures.",
    )
    parser.add_argument(
        "--no-align-gt",
        action="store_true",
        help="Disable Umeyama alignment of ground-truth keyframes for logging.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep time between frames (seconds).",
    )
    parser.add_argument(
        "--perf-log-threshold",
        type=float,
        default=None,
        help="If set, log performance sections for non-keyframe frames whose total processing time "
        "exceeds this threshold (seconds).",
    )
    parser.add_argument(
        "--log-loop-closures",
        action="store_true",
        help="Log loop closures to console.",
    )
    parser.add_argument(
        "--matcher",
        choices=("lighterglue", "orb"),
        default="lighterglue",
        help="Feature matcher backend to use for both frontend and loop closure.",
    )
    parser.add_argument(
        "--use-vision-odometry",
        action="store_true",
        help="Estimate odometry via the selected matcher + PnP instead of using KITTI wheel odometry.",
    )
    parser.add_argument(
        "--use-gt-rotation",
        action="store_true",
        help="Override odometry rotation with the ground-truth rotation (with optional noise).",
    )
    parser.add_argument(
        "--gt-rotation-noise-deg",
        type=float,
        default=2.5,
        help="Std-dev for per-axis rotation noise applied to the GT-derived rotation (degrees).",
    )
    parser.add_argument(
        "--rotation-noise-deg",
        type=float,
        default=0.15,
        help="Std-dev (degrees) for per-axis rotation noise added to odometry.",
    )
    parser.add_argument(
        "--translation-noise",
        type=float,
        default=0.1,
        help="Std-dev (meters) for per-axis translation noise added to odometry.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Maximum number of frames to process (after the first odometry measurement).",
    )
    return parser.parse_args()


def main() -> None:
    multiprocessing.set_start_method("spawn")

    args = parse_args()
    calibration = get_kitti_calibration(args.sequence)

    loop_noise_sigmas = np.array([np.deg2rad(0.2), np.deg2rad(0.2), np.deg2rad(0.2), 0.1, 0.1, 0.1]) # TODO: proper covariance calibration for two-view loop closures?

    # low noise on pitch and roll since those should be easy to calculate absolutely from the IMU
    # magnometer should also be availiable here
    simulated_odometry_noise_sigmas = np.array([np.deg2rad(0.01), np.deg2rad(args.rotation_noise_deg), np.deg2rad(0.01), args.translation_noise, args.translation_noise, args.translation_noise])
    odometry_noise_sigmas = simulated_odometry_noise_sigmas if not args.use_vision_odometry else loop_noise_sigmas

    config = SlamConfig(
        keyframe_translation_threshold=args.keyframe_translation,
        keyframe_rotation_threshold=float(np.deg2rad(args.keyframe_rotation)),
        loop_min_inliers=args.loop_min_inliers,
        loop_worker_count=args.loop_workers,
        rerun_tcp_address=args.rerun_tcp,
        use_huber_loss=not args.disable_huber,
        align_ground_truth=not args.no_align_gt,
        enable_rerun_logging=not args.disable_rerun,
        feature_matcher=args.matcher,
        proximity_max_translation=50.0,
        proximity_max_rotation=float(np.deg2rad(100.0)),
        rectify_inputs=False,
        odometry_noise_sigmas=odometry_noise_sigmas,
        loop_noise_sigmas=loop_noise_sigmas, 
    )

    slam = create_default_slam_system(
        calibration_matrix=calibration.K_left_rect,
        config=config,
    )

    vision_odometry: VisionOdometryEstimator | None = None
    if args.use_vision_odometry:
        initial_frame = _load_first_kitti_frame(args.sequence)
        vision_odometry = VisionOdometryEstimator(
            matcher_type=args.matcher,
            rectify_inputs=config.rectify_inputs,
        )
        vision_odometry.prime(initial_frame)

    rotation_noise_sigmas = odometry_noise_sigmas[:3]
    translation_noise_sigmas = odometry_noise_sigmas[3:]
    iterator = get_kitti_iterator_with_odometry(
        sequence_id=args.sequence,
        include_ground_truth=True,
        rotation_noise_sigmas=rotation_noise_sigmas,
        translation_noise_sigmas=translation_noise_sigmas,
    )
    gt_rotation_noise_sigma = float(np.deg2rad(args.gt_rotation_noise_deg))
    gt_rotation_noise_sigmas = np.full(3, max(gt_rotation_noise_sigma, 1e-6), dtype=float)

    keyframe_count = 0
    loop_closure_count = 0
    last_snapshot = None
    processed_frames = 0

    try:
        for frame, odometry, gt_pose in iterator:
            odometry_input = odometry
            vo_estimate: OdometryEstimate | None = None
            if vision_odometry is not None:
                vo_estimate = vision_odometry.estimate(frame)
                if vo_estimate.pose is not None:
                    odometry_input = vo_estimate.pose
                    print(f"[VisionOdom] Successfully estimated visual odometry")
                else:
                    print(f"[VisionOdom] Failed to estimate visual odometry, falling back to KITTI odometry")
                    odometry_input = odometry

            absolute_rotation = None
            absolute_rotation_noise = None
            if args.use_gt_rotation and gt_pose is not None:
                absolute_rotation = gt_pose.rotation()
                if gt_rotation_noise_sigma > 0.0:
                    noise = np.random.normal(0.0, gt_rotation_noise_sigma, 3)
                    absolute_rotation = absolute_rotation.compose(gtsam.Rot3.Expmap(noise)) # isn't this right order so it doesn't scale the noise with the magnitude of the pose translation?
                absolute_rotation_noise = gt_rotation_noise_sigmas

            result = slam.process_step(
                frame,
                odometry_input,
                ground_truth_pose=gt_pose,
                absolute_rotation=absolute_rotation,
                absolute_rotation_noise=absolute_rotation_noise,
            )
            processed_frames += 1
            last_snapshot = result.performance

            if (
                vo_estimate is not None
                and vo_estimate.pose is None
                and vo_estimate.failure_reason is not None
            ):
                print(
                    f"[VisionOdom] Fallback to KITTI odometry at frame {result.frame_index:04d}: "
                    f"{vo_estimate.failure_reason} "
                    f"(matches={vo_estimate.match_count}, inliers={vo_estimate.inlier_count})"
                )

            total_duration = result.performance.total_duration
            log_sections = result.is_keyframe or (
                args.perf_log_threshold is not None and total_duration >= args.perf_log_threshold
            )

            if result.is_keyframe:
                keyframe_count += 1

            if log_sections:
                section_breakdown = ", ".join(
                    f"{name}:{duration:.3f}s"
                    for name, duration in sorted(result.performance.sections.items())
                    if duration > 0.0
                )
                if result.is_keyframe:
                    preprocess = result.frontend_timings.total if result.frontend_timings else 0.0
                    msg = f"[Frame {result.frame_index:04d}] keyframe {keyframe_count:04d} "
                    msg += f"(preprocess {preprocess:.2f}s, total {total_duration:.2f}s)\n"
                    msg += f"    sections: {section_breakdown}\n"
                    if result.metrics:
                        msg += f"    ATE transl {result.metrics.translation_ate:.3f} m, "
                        msg += f"distance {result.metrics.total_distance:.3f} m, "
                        msg += f"{result.metrics.translation_ate_pct:.3f}%"
                        msg += f", rot {result.metrics.rotation_ate_deg:.3f} deg\n"
                    print(msg)

                    if not args.disable_rerun:
                        rr.set_time("frame", sequence=result.frame_index)
                        rr.log("logs", rr.TextLog(msg))
                else:
                    header = (
                        f"[Frame {result.frame_index:04d}] non-keyframe total {total_duration:.2f}s"
                    )
                    if args.perf_log_threshold is not None:
                        header += f" (threshold {args.perf_log_threshold:.2f}s)"
                    msg = f"{header}\n    sections: {section_breakdown}"
                    print(msg)

                    if not args.disable_rerun:
                        rr.set_time("frame", sequence=result.frame_index)
                        rr.log("logs", rr.TextLog(msg))

            if result.loop_closures_added:
                loop_closure_count += result.loop_closures_added

                if args.log_loop_closures:
                    print(
                        f"    Integrated {result.loop_closures_added} loop closures "
                        f"(total {loop_closure_count})"
                    )

            if args.frame_limit is not None and processed_frames >= args.frame_limit:
                break

            time.sleep(args.sleep)
    finally:
        slam.shutdown()
        print(
            f"Finished with {keyframe_count} keyframes and {loop_closure_count} loop closures."
        )
        if last_snapshot is not None and last_snapshot.rolling_stats:
            total_runtime = last_snapshot.cumulative_time
            avg_frame_time = last_snapshot.mean_frame_time
            fps = last_snapshot.mean_fps
            print(
                f"Processed {last_snapshot.frame_count} frames | total {total_runtime:.2f}s | "
                f"avg {avg_frame_time:.3f}s ({fps:.2f} fps)"
            )


if __name__ == "__main__":
    main()
