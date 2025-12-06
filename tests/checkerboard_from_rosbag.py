"""
Quick checkerboard detector that streams images from a ROS1 bag and renders them in the terminal.

Example:
    pixi run -e ros2 python tests/checkerboard_from_rosbag.py --bag /home/jetson/kalibr_data/bag.bag --take-every 10
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image
from term_image import disable_queries
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from rosbags.typesys import Stores, get_typestore
from term_image.image import BaseImage, KittyImage

BaseImage.forced_support = True # force support for all term image formats
KittyImage.set_render_method("WHOLE") # render the whole image at once

def parse_board_yaml(board_path: Path) -> tuple[int, int]:
    """Load Kalibr-style board.yaml and return (cols, rows)."""
    text = board_path.read_text()
    data: dict[str, Any] = {}
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except Exception:
        # Fallback simple parser to avoid a hard dependency on PyYAML.
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.split("#", 1)[0].strip()

    try:
        cols = int(data["targetCols"])
        rows = int(data["targetRows"])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not parse checkerboard size from {board_path}") from exc

    return cols, rows


def select_image_connection(reader: AnyReader, topic: str | None):
    """Pick an image connection, preferring the requested topic or /left if present."""
    image_types = {"sensor_msgs/msg/Image", "sensor_msgs/Image"}
    image_conns = [c for c in reader.connections if c.msgtype in image_types]
    if not image_conns:
        raise RuntimeError("No image topics found in bag.")

    if topic:
        for conn in image_conns:
            if conn.topic == topic:
                return conn
        available = ", ".join(sorted(c.topic for c in image_conns))
        raise RuntimeError(f"Image topic {topic!r} not found. Available: {available}")

    for preferred in ("/left", "/cam0/image_raw", "/camera/image_raw"):
        for conn in image_conns:
            if conn.topic == preferred:
                return conn

    return image_conns[0]


def header_timestamp(msg: Any) -> float:
    """Get a float timestamp from a ROS header, tolerating different field names."""
    stamp = getattr(msg, "header", None)
    if stamp is None or getattr(stamp, "stamp", None) is None:
        return 0.0
    stamp = stamp.stamp
    sec = getattr(stamp, "sec", getattr(stamp, "secs", 0))
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nanosecs", getattr(stamp, "nsec", getattr(stamp, "nsecs", 0))))
    return float(sec) + float(nsec) * 1e-9


def to_terminal_image(img_bgr: np.ndarray) -> KittyImage:
    """Convert a BGR OpenCV image to a term-image instance."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return KittyImage(pil_img, width=100)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect checkerboards inside a ROS1 bag and render frames in the terminal.")
    parser.add_argument("--bag", type=Path, default=Path("/home/jetson/kalibr_data/bag.bag"), help="Path to rosbag file.")
    parser.add_argument("--board", type=Path, default=None, help="Path to Kalibr board.yaml (defaults to board.yaml beside the bag).")
    parser.add_argument("--image-topic", default=None, help="Image topic to use. If omitted, the first image topic (prefers /left) is selected.")
    parser.add_argument("--take-every", type=int, default=5, help="Subsample images by taking every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed frames (0 = no limit).")
    args = parser.parse_args()

    disable_queries()  # Avoid terminal query round-trips that block in non-interactive sessions.

    if args.take_every < 1:
        parser.error("--take-every must be >= 1")
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")

    bag_path = args.bag.expanduser()
    board_path = args.board.expanduser() if args.board else bag_path.with_name("board.yaml")
    if not bag_path.exists():
        raise FileNotFoundError(bag_path)
    if not board_path.exists():
        raise FileNotFoundError(board_path)

    cols, rows = parse_board_yaml(board_path)
    print(f"Using checkerboard {cols}x{rows} (inner corners) from {board_path}")
    print(f"Bag: {bag_path}")

    typestore = get_typestore(Stores.ROS1_NOETIC)

    seen = 0
    displayed = 0
    detections = 0
    clear = "\033[2J\033[H"

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        img_conn = select_image_connection(reader, args.image_topic)
        print(f"Image topic: {img_conn.topic}")

        for _, _, rawdata in reader.messages(connections=[img_conn]):
            msg = reader.deserialize(rawdata, img_conn.msgtype)
            seen += 1
            if (seen - 1) % args.take_every != 0:
                continue

            cv_img = message_to_cvimage(msg)
            if cv_img.ndim == 2:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)

            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)

            vis = cv_img.copy()
            status = "not found"
            if found:
                detections += 1
                status = "detected"
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(vis, (cols, rows), corners_refined, True)

            ts = header_timestamp(msg)
            term_img = to_terminal_image(vis)

            print(clear, end="")
            print(f"Frame {seen} @ {ts:.3f}s | {status} ({detections} found so far)")
            print(term_img)

            displayed += 1
            if args.max_frames and displayed >= args.max_frames:
                break

    print(f"Done. Displayed {displayed} frames (saw {seen}), detections: {detections}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
