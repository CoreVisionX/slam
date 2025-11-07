"""Thin runner for the refactored stereo SLAM system."""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import rerun as rr

from slam import SlamConfig, create_default_slam_system
from tests.test_utils import get_tartanair_iterator_with_odometry, tartanair_calib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the stereo SLAM pipeline on TartanAir data.")
    parser.add_argument("--env", default="AbandonedFactory", help="TartanAir environment name.")
    parser.add_argument("--difficulty", default="easy", help="TartanAir difficulty level.")
    parser.add_argument("--traj", default="P001", help="TartanAir trajectory id.")
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
    return parser.parse_args()


def main() -> None:
    multiprocessing.set_start_method('spawn') # use spawn instead of fork to avoid issues with multiprocessing

    args = parse_args()
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
    )

    slam = create_default_slam_system(
        calibration_matrix=tartanair_calib.K_left_rect,
        config=config,
    )

    iterator = get_tartanair_iterator_with_odometry(
        env=args.env,
        difficulty=args.difficulty,
        traj=args.traj,
        include_ground_truth=True,
        rotation_noise_sigmas=np.array([np.deg2rad(0.3)] * 3),
        translation_noise_sigmas=np.array([0.007, 0.007, 0.007]),
    )

    keyframe_count = 0
    loop_closure_count = 0
    last_snapshot = None

    try:
        for frame, odometry, gt_pose in iterator:
            result = slam.process_step(frame, odometry, ground_truth_pose=gt_pose)
            last_snapshot = result.performance

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
            print("Rolling performance stats:")
            for line in last_snapshot.summary_lines():
                print(f"    {line}")


if __name__ == "__main__":
    main()
