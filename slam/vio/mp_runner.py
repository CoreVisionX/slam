import multiprocessing as mp
from multiprocessing import Event, Lock
from multiprocessing.shared_memory import SharedMemory  # benchmark shared memory to make sure there's no overhead
from pathlib import Path
import time

import numpy as np

from slam.vio.core import VIO


# TODO: fix the shared memory leaks
# TODO: a proper initialization sequence/routine for a second of the robot sitting still could help a lot, esp. with imu frames and gravity initialization


def _extract_imu_window(
    imu_ts_buf: np.ndarray,
    imu_acc_buf: np.ndarray,
    imu_gyro_buf: np.ndarray,
    head: int,
    size: int,
    capacity: int,
    start_ts: float,
    end_ts: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return IMU samples in (start_ts, end_ts] in chronological order."""
    if size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )

    tail = (head - size) % capacity
    if tail < head:
        ts_seq = imu_ts_buf[tail:head]
        acc_seq = imu_acc_buf[tail:head]
        gyro_seq = imu_gyro_buf[tail:head]
    else:
        ts_seq = np.concatenate((imu_ts_buf[tail:], imu_ts_buf[:head]))
        acc_seq = np.concatenate((imu_acc_buf[tail:], imu_acc_buf[:head]))
        gyro_seq = np.concatenate((imu_gyro_buf[tail:], imu_gyro_buf[:head]))

    mask = (ts_seq > start_ts) & (ts_seq <= end_ts)
    if not np.any(mask):
        return (
            np.empty(0, dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )

    imu_ts_arr = ts_seq[mask].copy()
    imu_acc_arr = acc_seq[mask].copy()
    imu_gyro_arr = gyro_seq[mask].copy()
    
    # sort by timestamp
    sort_idx = np.argsort(imu_ts_arr)
    imu_ts_arr = imu_ts_arr[sort_idx]
    imu_acc_arr = imu_acc_arr[sort_idx]
    imu_gyro_arr = imu_gyro_arr[sort_idx]

    return imu_ts_arr, imu_acc_arr, imu_gyro_arr


def async_vio_worker(
    vio_config_path: str | Path,
    timestamp_shm: SharedMemory,
    left_rect_shm: SharedMemory,
    right_rect_shm: SharedMemory,
    imu_ts_buffer_shm: SharedMemory,
    imu_acc_buffer_shm: SharedMemory,
    imu_gyro_buffer_shm: SharedMemory,
    imu_head,
    imu_size,
    max_imu_samples: int,
    reset_timestamp_shm: SharedMemory,
    reset_left_rect_shm: SharedMemory,
    reset_right_rect_shm: SharedMemory,
    reset_t_shm: SharedMemory,
    reset_R_shm: SharedMemory,
    reset_v_shm: SharedMemory,
    lock: Lock,
    ready: Event,
    reset_ready: Event,
    reset_done: Event,
):
    vio = VIO.from_config(vio_config_path)

    imu_ts_buffer = np.ndarray(shape=(max_imu_samples,), dtype=np.float64, buffer=imu_ts_buffer_shm.buf)
    imu_acc_buffer = np.ndarray(shape=(max_imu_samples, 3), dtype=np.float64, buffer=imu_acc_buffer_shm.buf)
    imu_gyro_buffer = np.ndarray(shape=(max_imu_samples, 3), dtype=np.float64, buffer=imu_gyro_buffer_shm.buf)

    last_frame_timestamp: float | None = None
    frame_count = 0
    start_time = None

    while True:
        # Wait for reset parameters
        reset_ready.wait()
        with lock:
            reset_ready.clear()

            reset_timestamp_arr = np.ndarray(shape=1, dtype=np.float64, buffer=reset_timestamp_shm.buf)
            reset_timestamp = float(reset_timestamp_arr[0])

            reset_left_arr = np.ndarray(
                shape=(vio.config.height, vio.config.width, 3),
                dtype=np.uint8,
                buffer=reset_left_rect_shm.buf,
            )
            reset_right_arr = np.ndarray(
                shape=(vio.config.height, vio.config.width, 3),
                dtype=np.uint8,
                buffer=reset_right_rect_shm.buf,
            )
            reset_t_arr = np.ndarray(shape=(3,), dtype=np.float64, buffer=reset_t_shm.buf)
            reset_R_arr = np.ndarray(shape=(3, 3), dtype=np.float64, buffer=reset_R_shm.buf)
            reset_v_arr = np.ndarray(shape=(3,), dtype=np.float64, buffer=reset_v_shm.buf)

            # clear frame ready flag during reset to avoid processing stale data
            ready.clear()

        # Copy reset data outside the lock
        vio.reset(
            timestamp=reset_timestamp,
            left_rect=np.array(reset_left_arr, copy=True),
            right_rect=np.array(reset_right_arr, copy=True),
            t=reset_t_arr.copy(),
            R=reset_R_arr.copy(),
            v=reset_v_arr.copy(),
        )
        last_frame_timestamp = reset_timestamp
        frame_count = 0
        start_time = time.perf_counter()
        reset_done.set()

        # Process frames until another reset arrives
        while True:
            if reset_ready.is_set():
                break

            # allow reset interrupt with periodic check
            if not ready.wait(timeout=0.01):
                continue
            if reset_ready.is_set():
                ready.clear()
                break

            with lock:
                if reset_ready.is_set():
                    ready.clear()
                    break

                # read timestamp
                timestamp_arr = np.ndarray(shape=1, dtype=np.float64, buffer=timestamp_shm.buf)
                timestamp = float(timestamp_arr[0])

                # read frame data and copy out of shared memory
                left_rect_arr = np.ndarray(
                    shape=(vio.config.height, vio.config.width, 3),
                    dtype=np.uint8,
                    buffer=left_rect_shm.buf,
                )
                right_rect_arr = np.ndarray(
                    shape=(vio.config.height, vio.config.width, 3),
                    dtype=np.uint8,
                    buffer=right_rect_shm.buf,
                )
                left_rect = np.array(left_rect_arr, copy=True)
                right_rect = np.array(right_rect_arr, copy=True)

                ready.clear()

            with lock:
                head = imu_head.value
                size = imu_size.value

            imu_ts_arr, imu_acc_arr, imu_gyro_arr = _extract_imu_window(
                imu_ts_buffer,
                imu_acc_buffer,
                imu_gyro_buffer,
                head=head,
                size=size,
                capacity=max_imu_samples,
                start_ts=last_frame_timestamp,
                end_ts=timestamp,
            )

            if imu_ts_arr.size == 0:
                print(f"[AsyncVIO] Low IMU sample count: {imu_ts_arr.size}")
                continue

            vio.process(
                timestamp=timestamp,
                left_rect=left_rect,
                right_rect=right_rect,
                imu_acc=imu_acc_arr,
                imu_gyro=imu_gyro_arr,
                imu_ts=imu_ts_arr,
            )
            last_frame_timestamp = timestamp
            frame_count += 1
            if frame_count % 10 == 0 and start_time is not None:
                elapsed = time.perf_counter() - start_time
                if elapsed > 0:
                    fps = frame_count / elapsed
                    print(f"[AsyncVIO] Total processed frames: {frame_count}, FPS: {fps:.2f}")

            if frame_count % 100 == 0:
                print(f"[AsyncVIO] Total distance traveled: {vio.get_distance_traveled():.2f}m")


class AsyncVIO:
    def __init__(
        self,
        vio_config_path: str | Path,
        max_imu_samples: int = 640,
        height: int = 480,
        width: int = 848,
    ):  # TODO: remove height and width and load config instead
        self.vio_config_path = vio_config_path
        self.max_imu_samples = max_imu_samples

        # allocate shared memory for frame and imu data
        frame_size = width * height * 3
        self.timestamp_shm = SharedMemory(create=True, size=8)
        self.left_rect_shm = SharedMemory(create=True, size=frame_size)
        self.right_rect_shm = SharedMemory(create=True, size=frame_size)
        self.imu_ts_buffer_shm = SharedMemory(create=True, size=8 * max_imu_samples)
        self.imu_acc_buffer_shm = SharedMemory(create=True, size=8 * max_imu_samples * 3)
        self.imu_gyro_buffer_shm = SharedMemory(create=True, size=8 * max_imu_samples * 3)
        self.imu_head = mp.Value("i", 0)
        self.imu_size = mp.Value("i", 0)

        # reset buffers
        self.reset_timestamp_shm = SharedMemory(create=True, size=8)
        self.reset_left_rect_shm = SharedMemory(create=True, size=frame_size)
        self.reset_right_rect_shm = SharedMemory(create=True, size=frame_size)
        self.reset_t_shm = SharedMemory(create=True, size=8 * 3)
        self.reset_R_shm = SharedMemory(create=True, size=8 * 9)
        self.reset_v_shm = SharedMemory(create=True, size=8 * 3)

        # setup synchronization primitives
        self.lock = Lock()
        self.ready = Event()
        self.reset_ready = Event()
        self.reset_done = Event()

        # start worker immediately; it will wait for reset parameters
        self.worker = mp.Process(
            target=async_vio_worker,
            args=(
                self.vio_config_path,
                self.timestamp_shm,
                self.left_rect_shm,
                self.right_rect_shm,
                self.imu_ts_buffer_shm,
                self.imu_acc_buffer_shm,
                self.imu_gyro_buffer_shm,
                self.imu_head,
                self.imu_size,
                self.max_imu_samples,
                self.reset_timestamp_shm,
                self.reset_left_rect_shm,
                self.reset_right_rect_shm,
                self.reset_t_shm,
                self.reset_R_shm,
                self.reset_v_shm,
                self.lock,
                self.ready,
                self.reset_ready,
                self.reset_done,
            ),
        )
        self.worker.start()

    def reset(
        self,
        timestamp: float,
        left_rect: np.ndarray,
        right_rect: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        v: np.ndarray,
    ):
        # reset IMU ring buffer state
        with self.lock:
            self.imu_head.value = 0
            self.imu_size.value = 0
            np.ndarray(shape=(self.max_imu_samples,), dtype=np.float64, buffer=self.imu_ts_buffer_shm.buf)[:] = 0
            np.ndarray(
                shape=(self.max_imu_samples, 3), dtype=np.float64, buffer=self.imu_acc_buffer_shm.buf
            )[:] = 0
            np.ndarray(
                shape=(self.max_imu_samples, 3), dtype=np.float64, buffer=self.imu_gyro_buffer_shm.buf
            )[:] = 0

            # write reset parameters
            reset_timestamp_arr = np.ndarray(shape=1, dtype=np.float64, buffer=self.reset_timestamp_shm.buf)
            reset_timestamp_arr[:] = timestamp
            reset_left_arr = np.ndarray(shape=left_rect.shape, dtype=left_rect.dtype, buffer=self.reset_left_rect_shm.buf)
            reset_right_arr = np.ndarray(shape=right_rect.shape, dtype=right_rect.dtype, buffer=self.reset_right_rect_shm.buf)
            reset_left_arr[:] = left_rect
            reset_right_arr[:] = right_rect
            reset_t_arr = np.ndarray(shape=(3,), dtype=np.float64, buffer=self.reset_t_shm.buf)
            reset_t_arr[:] = t
            reset_R_arr = np.ndarray(shape=(3, 3), dtype=np.float64, buffer=self.reset_R_shm.buf)
            reset_R_arr[:] = R
            reset_v_arr = np.ndarray(shape=(3,), dtype=np.float64, buffer=self.reset_v_shm.buf)
            reset_v_arr[:] = v

            # ensure no pending frame signal
            self.ready.clear()
            self.reset_done.clear()
            self.reset_ready.set()

        # wait for worker to apply reset
        self.reset_done.wait()

    def stop(self):
        self.worker.terminate()
        self.worker.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

        return False # don't suppress exceptions

    def process(self, timestamp: float, left_rect: np.ndarray, right_rect: np.ndarray):
        """Enqueue a frame; IMU data is pulled from the shared ring buffer."""
        return self.process_frame(timestamp=timestamp, left_rect=left_rect, right_rect=right_rect)

    def process_frame(self, timestamp: float, left_rect: np.ndarray, right_rect: np.ndarray):
        with self.lock:
            self._write_frame_shared_memory(timestamp, left_rect, right_rect)
            self.ready.set()

    def process_imu(self, imu_ts: np.ndarray, imu_acc: np.ndarray, imu_gyro: np.ndarray):
        """Append IMU samples to the ring buffer."""
        with self.lock:
            self._write_imu_ring_buffer(imu_ts, imu_acc, imu_gyro)

    def _write_frame_shared_memory(self, timestamp: float, left_rect: np.ndarray, right_rect: np.ndarray):
        # create numpy views into shared memory
        timestamp_arr = np.ndarray(shape=1, dtype=np.float64, buffer=self.timestamp_shm.buf)
        left_rect_arr = np.ndarray(shape=left_rect.shape, dtype=left_rect.dtype, buffer=self.left_rect_shm.buf)
        right_rect_arr = np.ndarray(shape=right_rect.shape, dtype=right_rect.dtype, buffer=self.right_rect_shm.buf)

        # write data to shared memory
        timestamp_arr[:] = timestamp
        left_rect_arr[:] = left_rect
        right_rect_arr[:] = right_rect

    def _write_imu_ring_buffer(self, imu_ts: np.ndarray, imu_acc: np.ndarray, imu_gyro: np.ndarray):
        imu_ts = np.asarray(imu_ts, dtype=np.float64).reshape(-1)
        imu_acc = np.asarray(imu_acc, dtype=np.float64)
        imu_gyro = np.asarray(imu_gyro, dtype=np.float64)

        if imu_ts.shape[0] != imu_acc.shape[0] or imu_ts.shape[0] != imu_gyro.shape[0]:
            raise ValueError("IMU timestamps, accelerations, and gyros must have matching lengths")
        if imu_acc.ndim != 2 or imu_gyro.ndim != 2 or imu_acc.shape[1] != 3 or imu_gyro.shape[1] != 3:
            raise ValueError("IMU acceleration and gyro arrays must have shape (N, 3)")

        ts_buf = np.ndarray(shape=(self.max_imu_samples,), dtype=np.float64, buffer=self.imu_ts_buffer_shm.buf)
        acc_buf = np.ndarray(shape=(self.max_imu_samples, 3), dtype=np.float64, buffer=self.imu_acc_buffer_shm.buf)
        gyro_buf = np.ndarray(shape=(self.max_imu_samples, 3), dtype=np.float64, buffer=self.imu_gyro_buffer_shm.buf)

        head = self.imu_head.value
        size = self.imu_size.value

        for ts, acc, gyro in zip(imu_ts, imu_acc, imu_gyro):
            idx = head % self.max_imu_samples
            ts_buf[idx] = ts
            acc_buf[idx] = acc
            gyro_buf[idx] = gyro
            head = (head + 1) % self.max_imu_samples
            if size < self.max_imu_samples:
                size += 1

        self.imu_head.value = head
        self.imu_size.value = size
