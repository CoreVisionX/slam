from slam.local_vo.bundle_adjustment import finite_difference_velocity
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm

from slam.vio.core import VIO
from slam.vio import save_tum_sequence
from tests.datasets.pipeline import SequencePreprocessor


# TODO: there should be an easy way to load VIO that doesn't require knowing anything about hydra. Just have a library function that uses compose with default or overriden config options and instantiates the VIO class in one easy line
@hydra.main(version_base=None, config_path="config", config_name="vio_config")
def main(cfg: DictConfig):
    # Instantiate VIO directly from config
    vio: VIO = instantiate(cfg.vio)
    
    # Load data pipeline
    pipeline: SequencePreprocessor = instantiate(cfg.data_pipeline, _convert_="partial")
    preprocessed = pipeline.prepare(seed=13, max_depth=40.0)
    rectified_frames = preprocessed.rectified_frames
    sequence = preprocessed.sequence
    
    print(f"Processing {len(rectified_frames)} frames...")

    # Reset VIO with initial state
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
    output_path = Path(__file__).parent / "results" / "vio_estimated.txt"
    save_tum_sequence(vio.get_estimated_trajectory(), str(output_path))

if __name__ == "__main__":
    main()
