"""Fixed operations in a disposable process, so timeouts can stop native model work."""
import json
import logging
from pathlib import Path
import sys
from visionlab.image_processing import decode_image
from visionlab.detection import detect_objects, ModelUnavailable
from visionlab.segmentation import remove_background
from visionlab.video_processing import analyze_video
from visionlab.schemas import VideoOptions


def write_json(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def run(operation: str, folder: Path):
    options = json.loads((folder / "options.json").read_text())
    if operation == "video":
        result = analyze_video(folder / options.pop("source"), folder / "output.avi",
                               VideoOptions(**options),
                               lambda p: write_json(folder / "progress.json", p))
    else:
        image = decode_image((folder / "input").read_bytes())
        if operation == "detect":
            output, result = detect_objects(image, **options)
        elif operation == "remove":
            output, cutout, result = remove_background(image, options.get("background"))
            cutout.save(folder / "transparent.png")
        else:
            raise ValueError("Unknown processing operation.")
        output.save(folder / "output.png")
    write_json(folder / "result.json", result)


if __name__ == "__main__":
    folder = Path(sys.argv[2])
    try:
        run(sys.argv[1], folder)
    except ModelUnavailable as exc:
        write_json(folder / "error.json", {"status": 503, "message": str(exc)})
        sys.exit(1)
    except ValueError as exc:
        write_json(folder / "error.json", {"status": 422, "message": str(exc)})
        sys.exit(1)
    except Exception:
        logging.exception("Processing worker failed")
        write_json(folder / "error.json", {"status": 500, "message": "Processing failed. Check the server terminal for details."})
        sys.exit(1)
