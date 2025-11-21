# %%
from pathlib import Path
import sys
import os

import numpy as np
import rerun as rr
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slam.local_vo import KLTFeatureTracker
from tests.datasets.pipeline import SequencePreprocessor


def _load_config(config_name: str = "klt"):
    config_dir = Path(__file__).parent / "config"
    with initialize_config_dir(config_dir=str(config_dir), job_name="klt_demo", version_base=None):
        return compose(config_name=config_name)


# %%
cfg = _load_config()
pipeline: SequencePreprocessor = instantiate(cfg.data_pipeline, _recursive_=False)
tracker: KLTFeatureTracker = instantiate(cfg.klt_tracker)

preprocessed = pipeline.prepare()
rectified = preprocessed.rectified_frames
if not preprocessed.depth_variants:
    raise RuntimeError("Data pipeline did not produce any depth frames for KLT demo.")
depth_frames = preprocessed.depth_variants[0].frames

tracker.reset()
rr.init("klt_tracks", spawn=True)

for idx, (frame, depth) in enumerate(zip(rectified, depth_frames)):
    observations = tracker.track_frame(frame, depth)

    rr.set_time("frame", sequence=idx)
    rr.log("image", rr.Image(frame.left_rect))
    if not observations:
        continue

    points = np.array([obs.keypoint for obs in observations.values()], dtype=np.float32)
    class_ids = [key for key in observations.keys()]
    rr.log("image/tracks", rr.Points2D(points, radii=3.0, class_ids=class_ids))

    n_observations = len(observations)
    rr.log("n_observations", rr.Scalars(n_observations))

if tracker.tracks:
    track_lengths = np.array([len(track.observations) for track in tracker.tracks.values()], dtype=np.int32)
    unique_lengths, counts = np.unique(track_lengths, return_counts=True)
    final_frame_idx = max(len(rectified) - 1, 0)
    rr.set_time("frame", sequence=final_frame_idx)
    rr.log(
        "track_statistics/track_length_histogram",
        rr.BarChart(values=counts.astype(np.int32), abscissa=unique_lengths.astype(np.int32)),
        static=True,
    )

# %%
