from collections.abc import Generator, Sequence
from pathlib import Path
import time

import numpy as np

from slam.hydra_utils import compose_config
from slam.vio.mp_runner import AsyncVIO
from slam.vio.types import VIOEstimate

try:
    from slam.vio import rs_sdk
except ImportError as err:
    raise ImportError(
        "rs_sdk not found. Please install librealsense with Python bindings to use the D435i helper."
    ) from err


class D435iVIO(AsyncVIO):
    """
    Thin convenience wrapper that owns the RealSense stream and the AsyncVIO worker.
    
    Typical usage:
        vio = D435iVIO()
        for estimate in vio:  # Ctrl+C to stop
            print(estimate.t)
    """

    def __init__(
        self,
        vio_config_path: str | Path = "config/vio_d435i.yaml",
        vio_overrides: Sequence[str] | None = None,
        fps: int = 30,
        init_frames: int = 60,
        warmup: float = 1.0,
        skip_frames: int = 100,
    ):
        self.vio_config_path = Path(vio_config_path)
        self._skip_frames = skip_frames
        self._stopped = False

        # Read VIO config first to know the camera resolution
        self.config = compose_config(self.vio_config_path, overrides=vio_overrides)
        width = int(self.config.config.width)
        height = int(self.config.config.height)

        # Start the RealSense stream
        self.stream = rs_sdk.D435iIterator(width=width, height=height, fps=fps)

        # Gather a short IMU window to seed gravity and gyro bias
        gravity, gyro_bias = self._estimate_initial_bias(init_frames=init_frames)

        overrides = list(vio_overrides) if vio_overrides is not None else []
        overrides.append(f"imu_preintegrator.config.gravity={gravity.tolist()}")
        overrides.append(f"imu_preintegrator.config.initial_gyro_bias={gyro_bias.tolist()}")

        super().__init__(vio_config_path=self.vio_config_path, vio_overrides=overrides)

        if warmup > 0:
            time.sleep(warmup)

        self._prime_stream_and_reset()

    def _estimate_initial_bias(self, init_frames: int) -> tuple[np.ndarray, np.ndarray]:
        imu_acc_samples: list[np.ndarray] = []
        imu_gyro_samples: list[np.ndarray] = []

        for i, (_t_curr, _left_rect, _right_rect, imu_ts, imu_gyro, imu_acc) in enumerate(self.stream):
            if imu_ts.size > 0:
                imu_acc_samples.extend(imu_acc.tolist())
                imu_gyro_samples.extend(imu_gyro.tolist())

            if i + 1 >= init_frames:
                break

        if not imu_acc_samples or not imu_gyro_samples:
            self.stream.close()
            raise RuntimeError("Failed to collect IMU samples from the D435i during initialization.")

        imu_acc_arr = np.asarray(imu_acc_samples, dtype=np.float64)
        imu_gyro_arr = np.asarray(imu_gyro_samples, dtype=np.float64)

        gravity = -np.mean(imu_acc_arr, axis=0)
        gyro_bias = np.mean(imu_gyro_arr, axis=0)

        return gravity, gyro_bias

    def _prime_stream_and_reset(self) -> None:
        try:
            for _ in range(self._skip_frames):
                next(iter(self.stream))
            first = next(iter(self.stream))
        except StopIteration:
            self.stop()
            raise RuntimeError("No frames received from RealSense. Is the D435i connected?")

        t0, left0, right0, _, _, _ = first

        super().reset(
            timestamp=float(t0),
            left_rect=left0,
            right_rect=right0,
            t=np.zeros(3, dtype=np.float64),
            R=np.eye(3, dtype=np.float64),
            v=np.zeros(3, dtype=np.float64),
        )

    def estimates(self) -> Generator[VIOEstimate, None, None]:
        """
        Continuously yield the latest VIO estimate while streaming from the D435i.
        
        Stops automatically when the stream ends or when the generator is closed.
        """
        try:
            for t_curr, left_rect, right_rect, imu_ts, imu_gyro, imu_acc in self.stream:
                self.process_imu(imu_ts=imu_ts, imu_acc=imu_acc, imu_gyro=imu_gyro)
                self.process(timestamp=float(t_curr), left_rect=left_rect, right_rect=right_rect)

                estimate = self.get_current_estimate()
                if estimate is not None:
                    yield estimate
        finally:
            self.stop()

    def __iter__(self):
        return self.estimates()

    def stop(self) -> None:
        if getattr(self, "_stopped", False):
            return
        self._stopped = True
        try:
            if hasattr(self, "stream") and self.stream is not None:
                self.stream.close()
        finally:
            try:
                super().stop()
            except Exception as err:
                # Shared memory unlink can raise if already cleaned up; ignore to keep shutdown smooth
                print(f"Error stopping D435iVIO: {err}")
