"""Install the optional CPU stack and prepare weights before opening the studio."""
import argparse
import subprocess
import sys

from .config import ROOT
from .platform_support import model_install_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true", help="Skip dependency installation.")
    parser.add_argument("--small", action="store_true", help="Also prepare the larger YOLO11s detector.")
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] <= (3, 12):
        parser.error("Use Python 3.11 or 3.12 in the project's virtual environment.")
    try:
        if not args.download_only:
            command = model_install_command()
            subprocess.run(command, check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements-models.txt")], check=True)
        from .detection import load_detector
        from .segmentation import load_session
        print("Preparing YOLO11n. The first download needs an internet connection.", flush=True)
        load_detector("yolo11n")
        if args.small:
            print("Preparing YOLO11s…", flush=True)
            load_detector("yolo11s")
        print("Preparing the background-removal model…", flush=True)
        load_session()
        print("Models are ready. Restart VisionLab if it is already running.")
        return 0
    except (subprocess.CalledProcessError, RuntimeError, ImportError, OSError, ValueError) as exc:
        print(f"Model setup did not finish: {exc}", file=sys.stderr)
        print("Check your connection and retry. Existing downloaded weights are kept.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
