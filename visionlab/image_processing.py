"""Small, non-mutating operations on Pillow images."""
from io import BytesIO
import warnings
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from .config import IMAGE_FORMATS, MAX_IMAGE_BYTES, MAX_PIXELS, MAX_IMAGE_SIDE
from .schemas import Adjustments

Image.MAX_IMAGE_PIXELS = MAX_PIXELS
cv2.setNumThreads(2)


def decode_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Choose an image smaller than 12 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format not in IMAGE_FORMATS:
                    raise ValueError("Use a PNG, JPEG, or WebP image.")
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError("Images must contain at most 12 million pixels.")
                if max(source.size) > MAX_IMAGE_SIDE:
                    raise ValueError("Each image side must be at most 8,192 pixels.")
                source.load()
                return ImageOps.exif_transpose(source).convert("RGBA")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("This image is damaged, unsupported, or too large.") from exc


def encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def process_image(image: Image.Image, options: Adjustments) -> Image.Image:
    # Keep the source alpha channel through RGB-only OpenCV operations.
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    result = rgba.convert("RGB")
    for enhancer, value in ((ImageEnhance.Brightness, options.brightness),
                            (ImageEnhance.Contrast, options.contrast),
                            (ImageEnhance.Color, options.saturation),
                            (ImageEnhance.Sharpness, options.sharpness)):
        result = enhancer(result).enhance(value)
    if options.grayscale:
        result = ImageOps.grayscale(result).convert("RGB")
    pixels = np.asarray(result)
    if options.blur:
        kernel = options.blur * 2 + 1
        pixels = cv2.GaussianBlur(pixels, (kernel, kernel), 0)
    if options.mode != "none":
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        if options.mode == "edges":
            pixels = cv2.Canny(gray, 80, 160)
        else:
            _, pixels = cv2.threshold(gray, options.threshold, 255, cv2.THRESH_BINARY)
    result = Image.fromarray(pixels).convert("RGBA")
    result.putalpha(alpha)
    return result
