from typing import Literal
from slam.viz import log_scalar
import rerun as rr
import numpy as np
import gtsam
from typing import Sequence, Mapping

from slam.registration.registration import RectifiedStereoFrame, StereoDepthFrame
from slam.viz import rr_log_pose, rr_log_trajectory
from .klt_tracker import TrackObservation, TrackObservationsBatch
from .types import VIOEstimate

class VIORerunLogger:
    def __init__(
        self,
        *,
        app_id: str = "vio_example",
        base_path: str = "vio",
        spawn: bool | None = True,
        url: str | None = None,
        # TODO: support more view coordinate conventions
        cam_view_coordinates: Literal["RDF", "RIGHT_HAND_X_UP", "RIGHT_HAND_Z_UP"] = "RDF",
        world_view_coordinates: Literal["RDF", "RIGHT_HAND_X_UP", "RIGHT_HAND_Z_UP"] = "RDF",
        trajectory_thickness: float = 0.008,
        trajectory_color: tuple[int, int, int] = (0, 0, 255),
        feature_radii: float = 3.0,
        image_plane_dist: float = 2.0,
    ) -> None:
        rr.init(app_id, spawn=spawn)

        if url:
            rr.connect_grpc(url)

        self._base_path = base_path.rstrip("/")
        if not self._base_path:
            self._base_path = "vio"

        self.trajectory_thickness = trajectory_thickness
        self.trajectory_color = trajectory_color
        self.feature_radii = feature_radii
        self.image_plane_dist = image_plane_dist

        # setup view coordinates
        view_coordinates = {
            "RDF": rr.ViewCoordinates.RDF,
            "RIGHT_HAND_X_UP": rr.ViewCoordinates.RIGHT_HAND_X_UP,
            "RIGHT_HAND_Z_UP": rr.ViewCoordinates.RIGHT_HAND_Z_UP,
        }

        self.cam_view_coordinates = view_coordinates[cam_view_coordinates]
        self.world_view_coordinates = view_coordinates[world_view_coordinates]

        rr.log("/", rr.ViewCoordinates(self.world_view_coordinates), static=True)

    def log_step(
        self,
        *,
        frame_idx: int,
        timestamp: float,
        pose: gtsam.Pose3,
        frame: RectifiedStereoFrame | StereoDepthFrame,
        trajectory: Sequence[gtsam.Pose3] | None,
        observations: Mapping[int, TrackObservation] | None,
        landmarks: Sequence[Mapping[str, object]] | None,
        all_landmarks: Sequence[Mapping[str, object]] | None = None,
        ba_stats: Mapping[str, int] | None = None,
        bias: gtsam.imuBias.ConstantBias | None = None,
        bias_trajectory: np.ndarray | None = None,
    ) -> None:
        """Log the current VIO step to rerun."""
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("timestamp", timestamp=timestamp)
        self._log_pose(pose, frame)
        self._log_trajectory(trajectory)
        self._log_klt_features(observations)
        self._log_landmarks(landmarks, f"{self._base_path}/landmarks/active")
        self._log_landmarks(all_landmarks, f"{self._base_path}/landmarks/all")
        self._log_bundle_stats(ba_stats)
        self._log_bias(bias)
        self._log_bias_trajectory(bias_trajectory)

        # log right rect
        rr.log("vio/pose/rgb_right", rr.Image(frame.right_rect))

    def _log_pose(self, pose: gtsam.Pose3, frame: RectifiedStereoFrame | StereoDepthFrame) -> None:
        rr_log_pose(f"{self._base_path}/pose", pose, frame, camera_xyz=self.cam_view_coordinates, image_plane_dist=self.image_plane_dist)

    def _log_trajectory(self, trajectory: Sequence[gtsam.Pose3] | None) -> None:
        if not trajectory:
            return
            
        rr_log_trajectory(f"{self._base_path}/trajectory", list(trajectory), radii=self.trajectory_thickness, color=self.trajectory_color)

    def _log_klt_features(self, observations: Mapping[int, TrackObservation] | TrackObservationsBatch | None) -> None:
        if not observations:
            return

        image_path = f"{self._base_path}/pose/rgb"
        if isinstance(observations, TrackObservationsBatch):
            points = observations.keypoints
            class_ids = observations.ids.tolist()
        else:
            obs_items = list(observations.items())
            points = np.array([obs.keypoint for _, obs in obs_items], dtype=np.float32)
            class_ids = [track_id for track_id, _ in obs_items]
        observation_count = len(class_ids)

        rr.log(
            image_path,
            rr.Points2D(points, radii=self.feature_radii, class_ids=class_ids),
        )
        rr.log(
            f"{self._base_path}/klt/observations",
            rr.Scalars(observation_count),
        )

    def _log_landmarks(self, landmarks: Sequence[Mapping[str, object]] | None, base_path: str) -> None:
        if not landmarks:
            rr.log(base_path, rr.Points3D(np.empty((0, 3), dtype=np.float32)))
            return

        positions = np.array([lm["position"] for lm in landmarks], dtype=np.float32)
        class_ids = [int(lm.get("original_track_id", lm.get("landmark_id", -1))) for lm in landmarks]

        rr.log(
            base_path,
            rr.Points3D(positions, class_ids=class_ids, radii=0.01),
        )

    def _log_bundle_stats(self, stats: Mapping[str, int] | None) -> None:
        if not stats:
            return

        base_path = f"{self._base_path}/bundle_adjustment"
        for key in stats.keys():
            rr.log(f"{base_path}/{key}", rr.Scalars(int(stats[key])))

    def _log_bias(self, bias: gtsam.imuBias.ConstantBias | None) -> None:
        if not bias:
            return

        acc_base_path = "accel_bias"
        gyro_base_path = "gyro_bias"

        log_scalar(f"{acc_base_path}/x", bias.accelerometer()[0])
        log_scalar(f"{acc_base_path}/y", bias.accelerometer()[1])
        log_scalar(f"{acc_base_path}/z", bias.accelerometer()[2])

        log_scalar(f"{gyro_base_path}/x", bias.gyroscope()[0])
        log_scalar(f"{gyro_base_path}/y", bias.gyroscope()[1])
        log_scalar(f"{gyro_base_path}/z", bias.gyroscope()[2])

    # TODO: doesn't really work correctly right now
    def _log_bias_trajectory(self, bias_trajectory: np.ndarray | None) -> None:
        if not bias_trajectory:
            return

        acc_base_path = "accel_bias_trajectory"
        gyro_base_path = "gyro_bias_trajectory"

        ts = bias_trajectory[:, 0]
        acc = bias_trajectory[:, 1:4]
        gyro = bias_trajectory[:, 4:]

        acc_x = acc[:, 0]
        acc_y = acc[:, 1]
        acc_z = acc[:, 2]

        gyro_x = gyro[:, 0]
        gyro_y = gyro[:, 1]
        gyro_z = gyro[:, 2]

        rr.log(f"{acc_base_path}/x", rr.LineStrips2D([np.stack([ts, acc_x], axis=1)]))
        rr.log(f"{acc_base_path}/y", rr.LineStrips2D([np.stack([ts, acc_y], axis=1)]))
        rr.log(f"{acc_base_path}/z", rr.LineStrips2D([np.stack([ts, acc_z], axis=1)]))

        rr.log(f"{gyro_base_path}/x", rr.LineStrips2D([np.stack([ts, gyro_x], axis=1)]))
        rr.log(f"{gyro_base_path}/y", rr.LineStrips2D([np.stack([ts, gyro_y], axis=1)]))
        rr.log(f"{gyro_base_path}/z", rr.LineStrips2D([np.stack([ts, gyro_z], axis=1)]))


def save_tum_sequence(vio_outputs: list[VIOEstimate], output_path: str):
    """Save VIO trajectory in TUM format (timestamp tx ty tz qx qy qz qw)."""
    if not vio_outputs:
        print("No VIO outputs to save")
        return
    
    rows = []
    for output in vio_outputs:
        # Convert rotation matrix to quaternion
        rot = gtsam.Rot3(output.R)
        quat = rot.toQuaternion()
        
        rows.append([
            output.timestamp,
            float(output.t[0]),
            float(output.t[1]),
            float(output.t[2]),
            float(quat.x()),
            float(quat.y()),
            float(quat.z()),
            float(quat.w()),
        ])
    
    tum_data = np.asarray(rows, dtype=np.float64)
    np.savetxt(output_path, tum_data, fmt="%.9f")
    print(f"Saved TUM trajectory: {output_path}")
