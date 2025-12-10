"""
Minimal Zenoh subscriber for D435i VIO position estimates.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable, Sequence

from pydantic.dataclasses import dataclass
import zenoh

DEFAULT_KEY_EXPR = "slam/vio/d435i"


@dataclass
class VIOEstimate:
    timestamp: float
    t: list[float]
    R: list[list[float]]
    v: list[float]


def _configure_session(endpoints: Sequence[str] | None) -> zenoh.Config:
    config = zenoh.Config()
    if endpoints:
        endpoints_json = json.dumps(list(endpoints))
        if hasattr(config, "insert_json5"):
            config.insert_json5("connect/endpoints", endpoints_json)
        else:
            config.insert("connect/endpoints", endpoints_json)  # type: ignore[arg-type]
    return config


def _init_zenoh_logging() -> None:
    if hasattr(zenoh, "init_log_from_env"):
        zenoh.init_log_from_env()
    elif hasattr(zenoh, "init_log_from_env_or"):
        zenoh.init_log_from_env_or("info")
    elif hasattr(zenoh, "try_init_log_from_env"):
        zenoh.try_init_log_from_env()


class D435iClient:
    def __init__(self, key_expr: str = DEFAULT_KEY_EXPR, connect: Sequence[str] | None = None):
        _init_zenoh_logging()
        self._key_expr = key_expr
        self._config = _configure_session(connect)
        self._session = zenoh.open(self._config)
        self._subs: list[zenoh.Subscriber] = []
        self._closed = False

    def subscribe(self, handler: Callable[[VIOEstimate], None]) -> Callable[[], None]:
        if self._closed:
            raise RuntimeError("Client is already closed.")

        def _listener(sample: zenoh.Sample) -> None:
            raw = sample.payload.to_bytes() if hasattr(sample.payload, "to_bytes") else bytes(sample.payload)
            payload = json.loads(raw.decode("utf-8"))
            estimate = VIOEstimate(**payload)
            handler(estimate)

        subscriber = self._session.declare_subscriber(self._key_expr, _listener)
        self._subs.append(subscriber)
        return lambda: self._unsubscribe(subscriber)

    def _unsubscribe(self, subscriber: zenoh.Subscriber) -> None:
        try:
            subscriber.undeclare()
        finally:
            if subscriber in self._subs:
                self._subs.remove(subscriber)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for sub in list(self._subs):
            self._unsubscribe(sub)
        self._session.close()

    def __enter__(self) -> "D435iClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Subscribe to D435i VIO estimates over Zenoh.")
    parser.add_argument("--key", default=DEFAULT_KEY_EXPR, help="Zenoh key expression to subscribe to.")
    parser.add_argument(
        "-e",
        "--connect",
        action="append",
        help="Optional Zenoh endpoints to connect to (can be repeated).",
    )
    args = parser.parse_args(argv)

    def _printer(estimate: VIOEstimate) -> None:
        print(f"{estimate.timestamp:.2f}s -> position {estimate.t}")

    with D435iClient(key_expr=args.key, connect=args.connect) as client:
        client.subscribe(_printer)
        print(f"Listening for VIO estimates on '{args.key}' (Ctrl+C to stop)...")
        
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Stopping subscriber...")
        finally:
            client.close()


if __name__ == "__main__":
    main()
