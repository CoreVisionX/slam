"""Performance timing utilities for the SLAM pipeline."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import mean
import time
from typing import Deque, Iterator

import numpy as np


@dataclass(slots=True)
class SectionStats:
    name: str
    latest: float
    average: float
    p95: float
    max_: float


@dataclass(slots=True)
class PerformanceSnapshot:
    frame_index: int
    total_duration: float
    sections: dict[str, float]
    rolling_stats: dict[str, SectionStats]
    frame_count: int
    cumulative_time: float

    def summary_lines(self) -> list[str]:
        lines = []
        for name, stats in self.rolling_stats.items():
            lines.append(
                f"{name}: latest {stats.latest:.3f}s | avg {stats.average:.3f}s | "
                f"p95 {stats.p95:.3f}s | max {stats.max_:.3f}s"
            )
        return lines

    @property
    def mean_frame_time(self) -> float:
        if self.frame_count == 0:
            return 0.0
        return self.cumulative_time / self.frame_count

    @property
    def mean_fps(self) -> float:
        mean_time = self.mean_frame_time
        if mean_time <= 0.0:
            return 0.0
        return 1.0 / mean_time


class PerformanceTracker:
    """Track per-section timings with rolling statistics."""

    def __init__(self, history: int = 200) -> None:
        self._history = history
        self._records: dict[str, Deque[float]] = {}
        self._step_sections: dict[str, float] = {}
        self._current_frame: int = -1
        self._frame_count = 0
        self._cumulative_time = 0.0

    def start_step(self, frame_index: int) -> None:
        self._step_sections = {}
        self._current_frame = frame_index

    def end_step(self, total_duration: float) -> PerformanceSnapshot:
        stats = {
            name: self._section_stats(name, deque_values)
            for name, deque_values in self._records.items()
            if deque_values
        }
        self._frame_count += 1
        self._cumulative_time += total_duration
        return PerformanceSnapshot(
            frame_index=self._current_frame,
            total_duration=total_duration,
            sections=dict(self._step_sections),
            rolling_stats=stats,
            frame_count=self._frame_count,
            cumulative_time=self._cumulative_time,
        )

    def record(self, section: str, duration: float) -> None:
        if section not in self._records:
            self._records[section] = deque(maxlen=self._history)
        self._records[section].append(duration)
        self._step_sections[section] = self._step_sections.get(section, 0.0) + duration

    @contextmanager
    def time_section(self, section: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.record(section, end - start)

    def _section_stats(self, name: str, values: Deque[float]) -> SectionStats:
        latest = values[-1]
        avg = mean(values)
        p95 = float(np.percentile(values, 95)) if len(values) > 1 else latest
        max_ = max(values)
        return SectionStats(name=name, latest=latest, average=avg, p95=p95, max_=max_)
