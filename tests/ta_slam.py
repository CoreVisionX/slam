"""Thin runner for the refactored stereo SLAM system."""

from __future__ import annotations

import argparse
import time
import multiprocessing    
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

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
        default=0.2,
        help="Sleep time between frames (seconds).",
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

    # TODO: collect (or build in tools for this?) metrics for timing performance, kinda important
    # figure out what's causing the stuttering here

    try:
        for frame, odometry, gt_pose in iterator:
            start_time = time.perf_counter()
            result = slam.process_step(frame, odometry, ground_truth_pose=gt_pose)
            elapsed = time.perf_counter() - start_time

            if result.is_keyframe:
                keyframe_count += 1
                preprocess = result.frontend_timings.total if result.frontend_timings else 0.0
                print(
                    f"[Frame {result.frame_index:04d}] keyframe {keyframe_count:04d} "
                    f"(preprocess {preprocess:.2f}s, total {elapsed:.2f}s)"
                )
                if result.metrics:
                    print(
                        f"    ATE transl {result.metrics.translation_ate:.3f} m, "
                        f"rot {result.metrics.rotation_ate_deg:.3f} deg"
                    )

            if result.loop_closures_added:
                loop_closure_count += result.loop_closures_added
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


if __name__ == "__main__":
    main()
