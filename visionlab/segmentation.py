"""Run the u2netp foreground model directly with ONNX Runtime.

Weights are the public u2netp export distributed by rembg. Using ONNX directly
keeps the app to one OpenCV installation and avoids importing a second CV stack.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import re
import tempfile
from time import perf_counter
import urllib.request

import numpy as np
from PIL import Image, ImageChops
from .detection import ModelUnavailable

MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
MODEL_SHA256 = "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"


def model_path() -> Path:
    return Path.home() / ".cache" / "visionlab" / "u2netp.onnx"


def has_expected_checksum(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == MODEL_SHA256


def prepare_weights() -> Path:
    destination = model_path()
    if has_expected_checksum(destination):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".download", delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 10 * 1024 * 1024:
                        raise ValueError("Unexpected background model download size.")
                    stream.write(chunk)
        if not has_expected_checksum(temporary):
            raise ValueError("The downloaded model checksum did not match. Please retry.")
        temporary.replace(destination)
        return destination
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def load_session():
    try:
        import onnxruntime as ort
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = 2
        settings.inter_op_num_threads = 1
        return ort.InferenceSession(str(prepare_weights()), sess_options=settings,
                                    providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ModelUnavailable(
            "Background removal could not load. Run python -m visionlab.setup_models "
            "to install requirements-models.txt and download verified u2netp weights."
        ) from exc


def foreground_mask(image: Image.Image, session) -> Image.Image:
    resized = image.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float32)
    # U²-Net's preprocessing divides by the image maximum. Black images need a guard.
    pixels /= max(float(pixels.max()), 1.0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1)[None])
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    prediction = np.asarray(output[0, 0], dtype=np.float32)
    if not np.isfinite(prediction).all():
        raise RuntimeError("The segmentation model returned a non-finite mask.")
    low, high = float(prediction.min()), float(prediction.max())
    if high - low > 1e-8:
        prediction = (prediction - low) / (high - low)
    else:
        prediction = np.clip(prediction, 0, 1)
    mask = Image.fromarray((prediction * 255).clip(0, 255).astype(np.uint8))
    return mask.resize(image.size, Image.Resampling.LANCZOS)


def remove_background(image: Image.Image, background: str | None = None):
    if background is not None and not re.fullmatch(r"#[0-9a-fA-F]{6}", background):
        raise ValueError("Background must be a six-digit hex color, such as #182338.")
    session = load_session()
    start = perf_counter()
    source = image.convert("RGBA")
    visible = Image.alpha_composite(Image.new("RGBA", source.size, "white"), source)
    mask = foreground_mask(visible, session)
    cutout = source.copy()
    cutout.putalpha(ImageChops.multiply(mask, source.getchannel("A")))
    elapsed = round((perf_counter() - start) * 1000, 1)
    output = cutout
    if background:
        output = Image.alpha_composite(Image.new("RGBA", cutout.size, background), cutout)
    return output, cutout, {"model": "u2netp", "backend": "ONNX Runtime", "device": "cpu", "inference_ms": elapsed}
