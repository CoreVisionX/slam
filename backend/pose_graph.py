import cv2
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X, V, B

import logging

from registration.registration import FeatureFrame


class GtsamPoseGraph:
    # think about how you can offload as much bookkeeping and timing and whatnot for odom/slam uses cases into this
    # it shouldn't just be a thin gtsam wrapper, it should take care of all the annoying preprocessing
    # so users can focus on just giving it sensor readings and getting useful outputs
    #
    # this should work easily for both LIO and VIO
    #
    # For LIO you need to make sure that the points are transformed by the optimized pose
    # before being added to the map.
    # You also might need to compensate for the coordinate frame difference somehow,
    # preferrably that could be done to the accel and gyro values before they're fed into this?
    
    # also to make initialization easy just use the first reading (or mean of first few readings) as the gravity vector

    # TODO: do we need to incorporate timestamps more than is done here?
    # TODO: use velocity factors derived from the encoders

    def __init__(self, K, pim=None):
        self.pose_prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self.vel_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        self.K = K

        # maintain complete factor graph for logging/debugging
        self.graph = gtsam.NonlinearFactorGraph()
        self.frames: dict[int, FeatureFrame] = {}

        self.kf_idx = 0

        # setup imu preintegration
        # TODO: look into gtsam's IMU combined factors? maybe they do something better than the regular IMU factors
        self.pim = pim

        self.isam_params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(self.isam_params)

        # track incremental additions for the iSAM2 optimizer
        self.pending_factors = gtsam.NonlinearFactorGraph()
        self.pending_values = gtsam.Values()

        # latest estimate from the optimizer
        self.values = gtsam.Values()

        # keep an initial estimate for the IMU bias so we can add it when the first IMU factor arrives
        self._bias_initial = self.pim.biasHat() if self.pim is not None else None

        # assume the robot is stationary at the start of the trajectory
        self.add_stationary_prior(0)
        self._update_solver()

    # TODO: use a stored optimizer/params
    def optimize(self):
        self._update_solver(relinearize=True)

    def predict_pose(self):
       """Predict the current pose based on the pending IMU preintegration values."""

       # figure out how to compose imu preintegration values with the latest pose and velocity
       
       pass

    def process_imu(self, linear_acc, ang_vel, dt):
        assert self.pim is not None

        self.pim.integrateMeasurement(linear_acc, ang_vel, dt)

    # TODO: velocity constraints on the odometry
    # TODO: decouple the frame from the odometry processing so they can be processed at different rates
    def process_odometry(self, prev_to_latest, prev_to_latest_noise, frame: FeatureFrame):
        prev_idx = self.kf_idx
        next_idx = self.kf_idx + 1

        assert self._has_value(X(prev_idx))

        if prev_to_latest is None:
            logging.warning('an odometery pose delta was not passed to GtsamPosegraph.process_odometry')

        # chain the new pose from the latest pose (if it doesn't already exist)
        if not self._has_value(X(next_idx)):
            prev_pose = self.values.atPose3(X(prev_idx))
            initial_pose = prev_pose.compose(prev_to_latest) if prev_to_latest is not None else prev_pose
            self._ensure_initial_value(X(next_idx), initial_pose)

        # TODO: add an option to initialize the new pose's velocity using constant velocity (and add a factor?) without requiring an IMU
        if self.pim is not None and not self._has_value(V(next_idx)):
            if self._has_value(V(prev_idx)):
                prev_velocity = self.values.atVector(V(prev_idx))
            else:
                prev_velocity = np.zeros(3)
            self._ensure_initial_value(V(next_idx), prev_velocity)

        # add odometery factor
        if self.pim is None:
            assert prev_to_latest and prev_to_latest_noise
            
        if prev_to_latest is not None:
            self.append_pose_factor(prev_to_latest, prev_to_latest_noise)

        # add imu factor and reset preintegration
        if self.pim is not None:
            self.append_imu_factor(self.pim)
            self.pim.resetIntegration()

        self.frames[next_idx] = frame
        self._update_solver()

    def append_imu_factor(self, pim):
        prev_idx = self.kf_idx
        next_idx = self.kf_idx + 1

        assert self._has_value(X(prev_idx))
        assert self._has_value(X(next_idx))

        if self._bias_initial is not None and not self._has_value(B(0)):
            self._ensure_initial_value(B(0), self._bias_initial)

        factor = gtsam.ImuFactor(X(prev_idx), V(prev_idx), X(next_idx), V(next_idx), B(0), pim)
        self.graph.add(factor)
        self.pending_factors.add(factor)

    def append_pose_factor(self, prev_to_latest, noise):
        prev_idx = self.kf_idx
        next_idx = self.kf_idx + 1

        assert self._has_value(X(prev_idx))
        assert self._has_value(X(next_idx))

        self.add_between_pose_factor(prev_idx, next_idx, prev_to_latest, noise)

        self.kf_idx += 1

    def add_between_pose_factor(self, kf_i_idx, kf_j_idx, i_to_j, noise):
        assert self._has_value(X(kf_i_idx))
        assert self._has_value(X(kf_j_idx))

        factor = gtsam.BetweenFactorPose3(X(kf_i_idx), X(kf_j_idx), i_to_j, noise)
        self.graph.add(factor)
        self.pending_factors.add(factor)

    # see https://github.com/borglab/gtsam/blob/2b0fe1749e350104362bf92e8287a357abc2cd8a/gtsam/slam/doc/EssentialMatrixConstraint.ipynb
    def add_scale_free_between_pose_factor(self, kf_i_idx, kf_j_idx, E_i_to_j, pts_i, pts_j, K, noise):
        assert self._has_value(X(kf_i_idx))
        assert self._has_value(X(kf_j_idx))

        # convert to gtsam essential matrix
        _, R_cv, t_cv, _mask = cv2.recoverPose(E_i_to_j, pts_i, pts_j, K)
        R = gtsam.Rot3(R_cv)
        t_hat = gtsam.Unit3(*t_cv.ravel())
        E_i_to_j_gtsam = gtsam.EssentialMatrix(R, t_hat)

        factor = gtsam.EssentialMatrixConstraint(X(kf_i_idx), X(kf_j_idx), E_i_to_j_gtsam, noise)
        self.graph.add(factor)
        self.pending_factors.add(factor)

    def add_pose_prior(self, kf_idx, pose):
        key = X(kf_idx)
        if not self._has_value(key):
            self._ensure_initial_value(key, pose)

        factor = gtsam.PriorFactorPose3(key, pose, self.pose_prior_noise)
        self.graph.add(factor)
        self.pending_factors.add(factor)

    def add_vel_prior(self, kf_idx, vel):
        key = V(kf_idx)
        if not self._has_value(key):
            self._ensure_initial_value(key, vel)

        factor = gtsam.PriorFactorVector(V(kf_idx), np.zeros(3), self.vel_prior_noise)
        self.graph.add(factor)
        self.pending_factors.add(factor)

    def add_stationary_prior(self, kf_idx):
        self.add_pose_prior(kf_idx, gtsam.Pose3.Identity())

        if self.pim is not None:
            self.add_vel_prior(kf_idx, np.zeros(3))

    def get_latest_pose(self):
        return self.get_pose(self.kf_idx)

    def get_pose(self, kf_idx):
        return self.values.atPose3(X(kf_idx))

    def _has_value(self, key):
        return self.values.exists(key) or self.pending_values.exists(key)

    def _ensure_initial_value(self, key, value):
        if not self.pending_values.exists(key) and not self.values.exists(key):
            self.pending_values.insert(key, value)

    def _update_solver(self, relinearize=False):
        has_new_factors = self.pending_factors.size() > 0

        if has_new_factors:
            self.isam.update(self.pending_factors, self.pending_values)
            self.pending_factors = gtsam.NonlinearFactorGraph()
            self.pending_values = gtsam.Values()
            self.values = self.isam.calculateEstimate()
        elif relinearize:
            self.isam.update()
            self.values = self.isam.calculateEstimate()
