"""CLI entry point for live SLAM demo."""

from __future__ import annotations

import sys
from pathlib import Path

# Add tests directory to path to import live_slam
REPO_ROOT = Path(__file__).resolve().parents[1]
tests_path = REPO_ROOT / "tests"

if tests_path.exists():
    # Development mode: import from tests directory
    if str(tests_path.parent) not in sys.path:
        sys.path.insert(0, str(tests_path.parent))
    from tests.live_slam import main
else:
    # Installed mode: live_slam functionality should be here
    raise ImportError(
        "Live SLAM demo requires access to tests/live_slam.py. "
        "Please install in development mode or run from the repository root."
    )


def cli_main() -> None:
    """Entry point for slam-live command."""
    main()


if __name__ == "__main__":
    cli_main()
