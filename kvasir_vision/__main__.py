import argparse
from sys import exit

from .calibrate import main as calibrate
from .capture_images import main as capture_images
from .locate import main as locate
from .show_distortion import main as show_distortion


def main() -> None:
    argp = argparse.ArgumentParser(
        description="A script to calibrate a camera using ChAruCoBoard images."
    )

    subparsers = argp.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "capture_images",
        help="Capture images of the ChAruCoBoard for calibration.",
    )
    subparsers.add_parser(
        "calibrate",
        help="Calibrate the camera using the captured images.",
    )
    subparsers.add_parser(
        "show_distortion",
        help="Show the distortion of the camera.",
    )
    subparsers.add_parser(
        "locate",
        help="Locate the robot marker in the camera frame.",
    )
    args = argp.parse_args()

    match args.command:
        case "capture_images":
            capture_images()
        case "calibrate":
            calibrate()
        case "show_distortion":
            show_distortion()
        case "locate":
            locate()
        case _:
            argp.print_help()
            exit(1)


if __name__ == "__main__":
    main()
