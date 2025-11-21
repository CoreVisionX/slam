import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from slam.vio.bundle_adjustment import finite_difference_velocity
from slam.vio import save_tum_sequence
from slam.vio.core import VIO
from tests.datasets.pipeline import SequencePreprocessor, load_euroc_pipeline


def run_vio(vio_config_path: Path, data_config_path: Path, output_path: Path) -> None:
    vio: VIO = VIO.from_config(vio_config_path)

    pipeline: SequencePreprocessor = load_euroc_pipeline(data_config_path)
    preprocessed = pipeline.prepare(seed=13, max_depth=40.0)
    rectified_frames = preprocessed.rectified_frames
    sequence = preprocessed.sequence
    
    print(f"Processing {len(rectified_frames)} frames...")

    # Reset VIO with the initial ground truth state
    gt_pose = sequence.world_poses[0]
    next_gt_pose = sequence.world_poses[1]
    dt = sequence.frame_timestamps[1] - sequence.frame_timestamps[0]
    first_velocity = finite_difference_velocity(gt_pose, next_gt_pose, dt)
    
    vio.reset(
        timestamp=sequence.frame_timestamps[0],
        left_rect=rectified_frames[0].left_rect,
        right_rect=rectified_frames[0].right_rect,
        t=gt_pose.translation(),
        R=gt_pose.rotation().matrix(),
        v=first_velocity,
    )
    
    # Process each frame in the sequence
    for idx in tqdm(range(1, len(rectified_frames)), desc="Estimating trajectory"):
        batch = sequence.imu_measurements[idx]
        
        vio.process(
            timestamp=sequence.frame_timestamps[idx],
            left_rect=rectified_frames[idx].left_rect,
            right_rect=rectified_frames[idx].right_rect,
            imu_acc=batch.linear_accelerations, # (b, n), m/s^2
            imu_gyro=batch.angular_velocities, # (b, n), rad/s
            imu_ts=batch.timestamps, # (b,) s
    )
    
    # Save TUM trajectory
    save_tum_sequence(vio.get_estimated_trajectory(), str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results" / "vio_estimated.txt",
        help="Path to write the estimated TUM trajectory.",
    )
    args = parser.parse_args()

    config_dir = Path(__file__).parent / "config"
    run_vio(
        vio_config_path=config_dir / "vio.yaml",
        data_config_path=config_dir / "euroc_pipeline.yaml",
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
