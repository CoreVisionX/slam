import rerun as rr
import numpy as np
import gtsam
from typing import Sequence, Mapping

from slam.registration.registration import RectifiedStereoFrame, StereoDepthFrame
from slam.viz import rr_log_pose, rr_log_trajectory
from .klt_tracker import TrackObservation
from .types import VIOEstimate

class VIORerunLogger:
    def __init__(
        self,
        *,
        app_id: str = "vio_example",
        base_path: str = "vio",
        spawn: bool = True,
        view_coordinates: rr.ViewCoordinates = rr.ViewCoordinates.RIGHT_HAND_X_UP,
        trajectory_thickness: float = 0.008,
        trajectory_color: tuple[int, int, int] = (0, 0, 255),
        feature_radii: float = 3.0,
        image_plane_dist: float = 2.0,
    ) -> None:
        rr.init(app_id, spawn=spawn)

        self._base_path = base_path.rstrip("/")
        if not self._base_path:
            self._base_path = "vio"

        self.view_coordinates = view_coordinates
        self.trajectory_thickness = trajectory_thickness
        self.trajectory_color = trajectory_color
        self.feature_radii = feature_radii
        self.image_plane_dist = image_plane_dist

    def log_step(
        self,
        *,
        frame_idx: int,
        timestamp: float,
        pose: gtsam.Pose3,
        frame: RectifiedStereoFrame | StereoDepthFrame,
        trajectory: Sequence[gtsam.Pose3],
        observations: Mapping[int, TrackObservation],
        landmarks: Sequence[Mapping[str, object]] | None,
        ba_stats: Mapping[str, int] | None = None,
    ) -> None:
        """Log the current VIO step to rerun."""
        rr.set_time("frame", sequence=frame_idx)
        rr.set_time("timestamp", timestamp=timestamp)
        self._log_pose(pose, frame)
        self._log_trajectory(trajectory)
        self._log_klt_features(observations)
        self._log_landmarks(landmarks)
        self._log_bundle_stats(ba_stats)

    def _log_pose(self, pose: gtsam.Pose3, frame: RectifiedStereoFrame | StereoDepthFrame) -> None:
        rr_log_pose(f"{self._base_path}/pose", pose, frame, camera_xyz=self.view_coordinates, image_plane_dist=self.image_plane_dist)

    def _log_trajectory(self, trajectory: Sequence[gtsam.Pose3]) -> None:
        if not trajectory:
            return
        rr_log_trajectory(f"{self._base_path}/trajectory", list(trajectory), radii=self.trajectory_thickness, color=self.trajectory_color)

    def _log_klt_features(self, observations: Mapping[int, TrackObservation]) -> None:
        image_path = f"{self._base_path}/pose/rgb"
        observation_count = len(observations)

        if observation_count == 0:
            rr.log(image_path, rr.Points2D(np.empty((0, 2), dtype=np.float32)))
            rr.log(f"{self._base_path}/klt/observations", rr.Scalars(0))
            return

        obs_items = list(observations.items())
        points = np.array([obs.keypoint for _, obs in obs_items], dtype=np.float32)
        class_ids = [track_id for track_id, _ in obs_items]
        rr.log(
            image_path,
            rr.Points2D(points, radii=self.feature_radii, class_ids=class_ids),
        )
        rr.log(
            f"{self._base_path}/klt/observations",
            rr.Scalars(observation_count),
        )

    def _log_landmarks(self, landmarks: Sequence[Mapping[str, object]] | None) -> None:
        base_path = f"{self._base_path}/landmarks"

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
