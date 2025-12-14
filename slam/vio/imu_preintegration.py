from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import gtsam

@dataclass
class ImuPreintegrationConfig:
    gravity: tuple[float, float, float]
    
    # Noise parameters
    accel_noise: float
    gyro_noise: float
    accel_random_walk: float
    gyro_random_walk: float
    integration_noise: float

    # Bias
    initial_acc_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_gyro_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    
class ImuPreintegrator:
    """
    Handles IMU preintegration using GTSAM.
    """    

    def __init__(self, config: ImuPreintegrationConfig):
        self.config = config
        
        # Validate and convert config to numpy arrays if needed
        gravity = np.array(self.config.gravity)
        
        # Setup IMU params
        self.params = gtsam.PreintegrationCombinedParams.MakeSharedU(np.linalg.norm(gravity))
        self.params.n_gravity = gravity
        
        # Set covariances
        # Note: GTSAM expects covariance matrices, config provides sigmas
        accel_cov = np.eye(3) * (self.config.accel_noise ** 2)
        gyro_cov = np.eye(3) * (self.config.gyro_noise ** 2)
        accel_random_walk_cov = np.eye(3) * (self.config.accel_random_walk ** 2)
        gyro_random_walk_cov = np.eye(3) * (self.config.gyro_random_walk ** 2)
        integration_cov = np.eye(3) * (self.config.integration_noise ** 2)
        
        self.params.setAccelerometerCovariance(accel_cov)
        self.params.setGyroscopeCovariance(gyro_cov)
        self.params.setBiasAccCovariance(accel_random_walk_cov)
        self.params.setBiasOmegaCovariance(gyro_random_walk_cov)
        self.params.setIntegrationCovariance(integration_cov)
        
        # Initialize bias
        self.initial_bias = gtsam.imuBias.ConstantBias(
            biasAcc=np.array(self.config.initial_acc_bias).reshape(3, 1),
            biasGyro=np.array(self.config.initial_gyro_bias).reshape(3, 1),
        )
        self.current_bias = self.initial_bias
        
        # Initialize PIM
        self.pim = gtsam.PreintegratedCombinedMeasurements(self.params, self.current_bias)

    def reset(self, bias: gtsam.imuBias.ConstantBias | None = None) -> None:
        """
        Reset the preintegration with a new bias.
        If no bias is provided, resets with the current bias.
        """
        if bias is not None:
            self.current_bias = bias
        self.pim.resetIntegrationAndSetBias(self.current_bias)

    def integrate(self, linear_acceleration: np.ndarray, angular_velocity: np.ndarray, dt: float) -> None:
        """
        Integrate a single IMU measurement.
        """
        self.pim.integrateMeasurement(linear_acceleration, angular_velocity, dt)

    def integrate_batch(self, linear_accelerations: list[np.ndarray], angular_velocities: list[np.ndarray], dts: list[float]) -> None:
        """
        Integrate a batch of IMU measurements.
        """
        for acc, gyro, dt in zip(linear_accelerations, angular_velocities, dts):
            self.integrate(acc, gyro, dt)

    # TODO: something about this or how it's used is wrong
    def delta(self, prev_pose: gtsam.Pose3, prev_velocity: np.ndarray) -> gtsam.NavState:
        """
        Get the relative delta from IMU preintegration.
        """
        
        nav_state = gtsam.NavState(prev_pose, prev_velocity)
        predicted = self.pim.predict(nav_state, self.current_bias) # current bias before integration
        
        return predicted

    @property
    def preintegrated_measurements(self) -> gtsam.PreintegratedImuMeasurements:
        return self.pim
