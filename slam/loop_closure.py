"""Background loop-closure processing."""

from __future__ import annotations

from dataclasses import dataclass
import atexit
import multiprocessing as mp
from multiprocessing import Process, Queue, SimpleQueue
from queue import Empty
import time
from typing import Iterable

import numpy as np

from registration.lighterglue import LighterglueMatcher
from registration.registration import FeatureFrame, IndexedFramePair
from registration.utils import solve_pnp


@dataclass(slots=True)
class LoopClosureResult:
    first_idx: int
    second_idx: int
    rotation_matrix: np.ndarray
    translation: np.ndarray
    inlier_count: int


_STOP_TOKEN = "__LOOP_CLOSURE_STOP__"


def _loop_closure_worker(
    loop_closure_candidates_queue: SimpleQueue,
    loop_closures_queue: Queue,
    min_inlier_count: int,
    worker_id: int,
    verbose: bool = False,
) -> None:
    """Background worker that estimates relative poses for candidate keyframe pairs."""

    matcher = LighterglueMatcher(num_features=4096, compile=False, device="cuda", use_lighterglue_matching=True)
    print(f"[LoopClosureWorker {worker_id}] started")

    pending_candidates: list[IndexedFramePair[FeatureFrame]] = []
    processing_times: list[float] = []
    log_every = 20

    while True:
        # Pull all available candidates to keep latency low.
        while not loop_closure_candidates_queue.empty():
            candidate = loop_closure_candidates_queue.get()
            if candidate == _STOP_TOKEN:
                print(f"[LoopClosureWorker {worker_id}] received stop signal")
                return
            pending_candidates.append(candidate)

        if not pending_candidates:
            time.sleep(0.01)
            continue

        candidate = pending_candidates.pop()  # prefer most recent candidate
        start_time = time.perf_counter()

        try:
            matched_pair = matcher.match([candidate])[0]
            first_to_second, matched_pair = solve_pnp(matched_pair)
        except Exception as exc:  # broad catch for robustness
            if verbose:
                print(
                f"[LoopClosureWorker {worker_id}] failed to solve PnP for "
                f"({candidate.first_idx}, {candidate.second_idx}): {exc}"
            )
            continue

        inlier_count = len(matched_pair.matches)

        if inlier_count >= min_inlier_count:
            loop_closures_queue.put(
                LoopClosureResult(
                    first_idx=candidate.first_idx,
                    second_idx=candidate.second_idx,
                    rotation_matrix=first_to_second.rotation().matrix(),
                    translation=first_to_second.translation(),
                    inlier_count=inlier_count,
                )
            )
            if verbose:
                print(
                f"[LoopClosureWorker {worker_id}] queued closure "
                f"{candidate.first_idx}->{candidate.second_idx} with {inlier_count} inliers"
            )

        end_time = time.perf_counter()
        processing_times.append(end_time - start_time)

        if len(processing_times) >= log_every:
            mean_time = float(np.mean(processing_times))
            print(
                f"[LoopClosureWorker {worker_id}] mean processing time: "
                f"{mean_time:.2f}s ({1.0 / mean_time:.2f} fps)"
            )
            processing_times.clear()


class LoopClosureManager:
    """Manage loop-closure worker processes and queues."""

    def __init__(
        self,
        min_inlier_count: int,
        num_workers: int | None = None,
    ) -> None:
        if num_workers is None or num_workers < 1:
            available = mp.cpu_count()
            num_workers = 1 if available <= 1 else min(4, max(1, available // 2))

        self._min_inlier_count = min_inlier_count
        self._num_workers = num_workers

        self._candidates_queue: SimpleQueue = SimpleQueue()
        self._results_queue: Queue = Queue(maxsize=5000)
        self._processes: list[Process] = []
        self._started = False

        atexit.register(self.stop)

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def start(self) -> None:
        if self._started:
            return

        self._processes = []
        for worker_idx in range(self._num_workers):
            process = Process(
                target=_loop_closure_worker,
                args=(self._candidates_queue, self._results_queue, self._min_inlier_count, worker_idx),
                daemon=True,
            )
            process.start()
            self._processes.append(process)
        self._started = True
        print(f"[LoopClosureManager] started {self._num_workers} worker(s)")

    def submit_candidates(self, candidates: Iterable[IndexedFramePair[FeatureFrame]]) -> None:
        for candidate in candidates:
            self._candidates_queue.put(candidate)

    def poll_results(self, max_results: int | None = None) -> list[LoopClosureResult]:
        results: list[LoopClosureResult] = []
        if not self._started:
            return results

        while max_results is None or len(results) < max_results:
            try:
                result = self._results_queue.get_nowait()
            except Empty:
                break
            if isinstance(result, LoopClosureResult):
                results.append(result)
        return results

    def stop(self) -> None:
        if not self._started:
            return

        for _ in self._processes:
            self._candidates_queue.put(_STOP_TOKEN)

        for process in self._processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()

        self._processes.clear()
        self._started = False
        print("[LoopClosureManager] stopped")
