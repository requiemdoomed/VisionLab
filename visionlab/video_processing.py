"""Bounded sequential video decoding. Outputs contain sampled frames, without audio."""
from pathlib import Path
from collections.abc import Callable
import math
import cv2
from PIL import Image
from .config import MAX_DURATION, MAX_VIDEO_FRAMES, MAX_VIDEO_PIXELS
from .schemas import VideoOptions, Adjustments
from .image_processing import process_image
from .detection import detect_objects
import numpy as np


def inspect_video(capture: cv2.VideoCapture) -> dict:
    if not capture.isOpened():
        raise ValueError("Video cannot be decoded. Try an H.264 MP4 or MJPEG AVI.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if not all(math.isfinite(v) and v > 0 for v in (fps, count, width, height)):
        raise ValueError("Video has missing or invalid metadata.")
    if width < 2 or height < 2 or count < 1:
        raise ValueError("Video must have at least one frame and dimensions of at least 2 × 2.")
    if (count > MAX_VIDEO_FRAMES or count / fps > MAX_DURATION or
            width * height > MAX_VIDEO_PIXELS or max(width, height) > 1920 or min(width, height) > 1080):
        raise ValueError("Video limit: 60 seconds, 3,600 frames, and 1920 × 1080 pixels per frame.")
    return {"fps": fps, "frames": int(count), "width": int(width),
            "height": int(height), "duration_seconds": round(count / fps, 2)}


def analyze_video(source: Path, destination: Path, options: VideoOptions,
                  progress: Callable[[dict], None] | None = None) -> dict:
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        metadata = inspect_video(capture)
        planned = min(options.max_samples, math.ceil(metadata["frames"] / options.sample_every))
        # MJPEG/AVI is a portable OpenCV export; browsers may require a download.
        scale = min(1, 960 / max(metadata["width"], metadata["height"]))
        width = max(2, int(metadata["width"] * scale) // 2 * 2)
        height = max(2, int(metadata["height"] * scale) // 2 * 2)
        fps = metadata["fps"] / options.sample_every
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
        writable = writer.isOpened()
        samples = []
        for index in range(min(metadata["frames"], MAX_VIDEO_FRAMES)):
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[0] * frame.shape[1] > MAX_VIDEO_PIXELS:
                raise ValueError("Decoded video frame exceeds the pixel limit.")
            if index % options.sample_every:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if options.operation == "detect":
                image, result = detect_objects(image, options.confidence, model=options.model,
                                               image_size=options.image_size, iou=options.iou)
            else:
                image = process_image(image, Adjustments(mode="edges"))
                result = {"operation": "edges"}
            samples.append({"frame": index, "seconds": round(index / metadata["fps"], 3), **result})
            image.thumbnail((960, 960))
            if writable:
                pixels = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
                writer.write(cv2.resize(pixels, (width, height)))
            if len(samples) == 1:
                image.save(destination.with_suffix(".png"))
            if progress:
                progress({"completed": len(samples), "total": planned})
            if len(samples) >= planned:
                break
        if not samples:
            raise ValueError("No frames could be decoded from this video.")
        writer.release()
        writer = None
        output_ok = False
        if writable and destination.exists():
            check = cv2.VideoCapture(str(destination))
            try:
                output_ok = check.isOpened() and check.read()[0]
            finally:
                check.release()
        return {"metadata": metadata, "samples": samples, "processed_frames": len(samples),
                "planned_frames": planned, "partial_decode": len(samples) < planned,
                "export_size": [width, height], "export_fps": fps,
                "export_available": output_ok, "operation": options.operation,
                "export_note": "Sampled frames only; no audio. MJPEG AVI; open in VLC or another video player."
                if output_ok else "The local MJPEG encoder could not produce a readable video. Frame analysis is available."}
    finally:
        capture.release()
        if writer is not None:
            writer.release()
