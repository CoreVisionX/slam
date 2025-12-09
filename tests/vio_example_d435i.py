import argparse

from slam.vio.d435i import D435iVIO


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames (0 = run until Ctrl+C)")
    args = parser.parse_args(argv)

    vio = D435iVIO()
    print("Streaming D435i VIO estimates (Ctrl+C to stop)...")

    try:
        for idx, estimate in enumerate(vio):
            print(f"{estimate.timestamp:.2f}s -> position {estimate.t}")
            if args.frames and idx + 1 >= args.frames:
                break
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        vio.stop()


if __name__ == "__main__":
    main()
