"""Command-line entry point for the MediaPipe hand detector.

Run from the project root: python -m hand.train image.jpg --view back
"""
import sys
from acupoint.cli import main


if __name__ == "__main__":
    main(["hand", *sys.argv[1:]])
