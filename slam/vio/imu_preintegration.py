from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import gtsam

@dataclass
class ImuPreintegrationConfig:
    gravity: tuple[float, float, float]
    gravity_magnitude: float
    
    # Noise parameters
    accel_noise: float
    gyro_noise: float
    integration_noise: float
    
    # Priors
    bias_prior_sigmas: list[float]
    velocity_prior_sigma: float = 10.0

class ImuPreintegrator:
    """
    Handles IMU preintegration using GTSAM.
    """
    def __init__(self, config: ImuPreintegrationConfig):
        self.config = config
        
        # Validate and convert config to numpy arrays if needed
        gravity = np.array(self.config.gravity)
        
        # Setup IMU params
        self.params = gtsam.PreintegrationParams.MakeSharedU(np.linalg.norm(gravity))
        self.params.n_gravity = gravity
        
        # Set covariances
        # Note: GTSAM expects covariance matrices, config provides sigmas
        accel_cov = np.eye(3) * (self.config.accel_noise ** 2)
        gyro_cov = np.eye(3) * (self.config.gyro_noise ** 2)
        integration_cov = np.eye(3) * (self.config.integration_noise ** 2)
        
        self.params.setAccelerometerCovariance(accel_cov)
        self.params.setGyroscopeCovariance(gyro_cov)
        self.params.setIntegrationCovariance(integration_cov)
        
        # Initialize bias
        # TODO: Allow initializing with specific bias values if needed
        self.current_bias = gtsam.imuBias.ConstantBias()
        
        # Initialize PIM
        self.pim = gtsam.PreintegratedImuMeasurements(self.params, self.current_bias)

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

    @property
    def preintegrated_measurements(self) -> gtsam.PreintegratedImuMeasurements:
        return self.pim
