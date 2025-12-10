"""
Publish D435i VIO estimates over Zenoh.

Run with:
    pixi run python -m vio.d435i_server --config tests/config/vio_d435i.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, Sequence

import numpy as np
import zenoh

from slam.vio.d435i import D435iVIO
from slam.vio.types import VIOEstimate

DEFAULT_KEY_EXPR = "slam/vio/d435i"


def _estimate_to_bytes(estimate: VIOEstimate) -> bytes:
    """Convert a VIOEstimate to JSON bytes with list-backed arrays."""
    payload = {
        "timestamp": float(estimate.timestamp),
        "t": np.asarray(estimate.t, dtype=float).tolist(),
        "R": np.asarray(estimate.R, dtype=float).tolist(),
        "v": np.asarray(estimate.v, dtype=float).tolist(),
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _configure_session(endpoints: Sequence[str] | None) -> zenoh.Config:
    config = zenoh.Config()
    if endpoints:
        endpoints_json = json.dumps(list(endpoints))
        if hasattr(config, "insert_json5"):
            config.insert_json5("connect/endpoints", endpoints_json)
        else:
            # Fallback for older zenoh bindings
            config.insert("connect/endpoints", endpoints_json)  # type: ignore[arg-type]
    return config


def _init_zenoh_logging() -> None:
    """Handle zenoh logging API differences."""
    if hasattr(zenoh, "init_log_from_env"):
        zenoh.init_log_from_env()
    elif hasattr(zenoh, "init_log_from_env_or"):
        zenoh.init_log_from_env_or("info")
    elif hasattr(zenoh, "try_init_log_from_env"):
        zenoh.try_init_log_from_env()


def publish_estimates(
    vio: Iterable[VIOEstimate],
    publisher: zenoh.Publisher,
    max_frames: int = 0,
) -> None:
    for idx, estimate in enumerate(vio):
        publisher.put(_estimate_to_bytes(estimate))
        if max_frames and idx + 1 >= max_frames:
            break


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Publish D435i VIO estimates over Zenoh.")
    parser.add_argument("--key", default=DEFAULT_KEY_EXPR, help="Zenoh key expression to publish to.")
    parser.add_argument(
        "--config",
        default="config/vio_d435i.yaml",
        help="Path to VIO config file (Hydra YAML).",
    )
    parser.add_argument(
        "-e",
        "--connect",
        action="append",
        help="Optional Zenoh endpoints to connect to (can be repeated).",
    )
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames (0 = run until Ctrl+C).")
    args = parser.parse_args(argv)

    # Ensure optional logger URL interpolation does not fail if unset
    os.environ.setdefault("RERUN_URL", 'rerun+http://192.168.1.20:9876/proxy')

    _init_zenoh_logging()
    z_config = _configure_session(args.connect)
    session = zenoh.open(z_config)
    publisher = session.declare_publisher(args.key)

    vio = D435iVIO(vio_config_path=args.config)
    print(f"Publishing D435i VIO estimates to '{args.key}' (Ctrl+C to stop)...")

    try:
        publish_estimates(vio, publisher, max_frames=args.frames)
    except KeyboardInterrupt:
        print("Stopping VIO publisher...")
    finally:
        try:
            vio.stop()
        finally:
            publisher.undeclare()
            session.close()


if __name__ == "__main__":
    main()
