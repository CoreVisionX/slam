from slam.local_vo.bundle_adjustment import finite_difference_velocity
import gtsam
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
import numpy as np

from slam.vio import VIO, save_tum_sequence
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
    
    # Setup IMU Preintegration Params
    # TODO: move the imu related params and logic into their own dedicated IMU preintegrator class
    ba_config = vio.ba.config

    # gravity
    imu_params = gtsam.PreintegrationParams.MakeSharedU(np.linalg.norm(vio.config.gravity))
    imu_params.n_gravity = vio.config.gravity

    # noise
    imu_params.setAccelerometerCovariance(np.eye(3) * (ba_config.imu_accel_noise**2))
    imu_params.setGyroscopeCovariance(np.eye(3) * (ba_config.imu_gyro_noise**2))
    imu_params.setIntegrationCovariance(np.eye(3) * (ba_config.imu_integration_noise**2))

    bias = gtsam.imuBias.ConstantBias()

    pim = gtsam.PreintegratedImuMeasurements(imu_params, bias)
    
    print(f"Processing {len(rectified_frames)} frames...")

   # Reset VIO
    gt_pose = sequence.world_poses[0]
    
    # Estimate initial velocity
    next_gt_pose = sequence.world_poses[1]
    dt = sequence.frame_timestamps[1] - sequence.frame_timestamps[0]
    first_velocity = finite_difference_velocity(gt_pose, next_gt_pose, dt)
    
    vio.reset(
        timestamp=sequence.frame_timestamps[0],
        left_rect=rectified_frames[0].left_rect,
        right_rect=rectified_frames[0].right_rect,
        t=gt_pose.translation(),
        R=gt_pose.rotation().matrix(),
        v=first_velocity
    )
    
    # Process each frame in the sequence
    for idx in tqdm(range(1, len(rectified_frames)), desc="Estimating trajectory"):
        rect_frame = rectified_frames[idx]
        timestamp = sequence.frame_timestamps[idx]
        
        # Preintegrate IMU measurements
        batch = sequence.imu_measurements[idx]
        for i in range(len(batch)):
            pim.integrateMeasurement(
                batch.linear_accelerations[i],
                batch.angular_velocities[i],
                deltaT=batch.dts[i],
            )

        # Process frame
        output = vio.process(
            timestamp=timestamp,
            left_rect=rect_frame.left_rect,
            right_rect=rect_frame.right_rect,
            pim=pim
        )
        
        pim.resetIntegrationAndSetBias(vio.get_estimated_bias())
    
    # Save TUM trajectory
    output_path = Path(__file__).parent / "results" / "vio_estimated.txt"
    save_tum_sequence(vio.get_estimated_trajectory(), str(output_path))

if __name__ == "__main__":
    main()
