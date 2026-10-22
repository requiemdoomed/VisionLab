from io import BytesIO
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from visionlab.image_processing import decode_image, encode_png, process_image
from visionlab.schemas import Adjustments, VideoOptions
from visionlab.segmentation import remove_background


@pytest.fixture
def sample():
    pixels = np.zeros((48, 64, 4), dtype=np.uint8)
    pixels[:, :, 3] = 128
    pixels[12:36, 16:48, :3] = (240, 100, 50)
    return Image.fromarray(pixels)


@pytest.mark.parametrize('options', [dict(brightness=1.4), dict(contrast=2), dict(saturation=0),
                                    dict(sharpness=2), dict(grayscale=True), dict(blur=3),
                                    dict(mode='edges'), dict(mode='threshold')])
def test_operations_preserve_input_dimensions_and_alpha(sample, options):
    before = sample.tobytes()
    result = process_image(sample, Adjustments(**options))
    assert result.size == sample.size
    assert sample.tobytes() == before
    assert result.getchannel('A').tobytes() == sample.getchannel('A').tobytes()
    assert result.tobytes() != before


def test_identity_and_encoded_output(sample):
    result = process_image(sample, Adjustments())
    assert result.tobytes() == sample.tobytes()
    assert decode_image(encode_png(result)).tobytes() == result.tobytes()


def test_threshold_and_grayscale(sample):
    gray = np.array(process_image(sample, Adjustments(grayscale=True)))
    assert np.array_equal(gray[:, :, 0], gray[:, :, 1])
    binary = np.array(process_image(sample, Adjustments(mode='threshold')))
    assert set(np.unique(binary[:, :, :3])) <= {0, 255}


def test_black_brightness(sample):
    result = np.array(process_image(sample, Adjustments(brightness=0)))
    assert not result[:, :, :3].any()


@pytest.mark.parametrize('data', [b'', b'not an image', b'\x89PNG\r\n'])
def test_bad_images(data):
    with pytest.raises(ValueError):
        decode_image(data)


def test_unsupported_format(sample):
    stream = BytesIO()
    sample.save(stream, format='GIF')
    with pytest.raises(ValueError, match='PNG'):
        decode_image(stream.getvalue())


def test_pixel_limit(sample, monkeypatch):
    monkeypatch.setattr('visionlab.image_processing.MAX_PIXELS', 20)
    with pytest.raises(ValueError, match='pixels'):
        decode_image(encode_png(sample))


@pytest.mark.parametrize('options', [dict(brightness=-1), dict(contrast=4), dict(blur=21),
                                    dict(mode='invalid'), dict(threshold=256), dict(saturation=float('nan')),
                                    dict(sharpness=float('inf')), dict(blur=1.5), dict(unknown=1)])
def test_bad_parameters(options):
    with pytest.raises(ValidationError):
        Adjustments(**options)


def test_video_limits():
    with pytest.raises(ValidationError):
        VideoOptions(max_samples=61)


def test_invalid_color_checked_before_model_load(sample):
    with pytest.raises(ValueError, match='hex'):
        remove_background(sample, 'red')


@pytest.mark.parametrize('fps,frames,width,height', [
    (0, 20, 640, 480), (float('nan'), 20, 640, 480),
    (30, 1900, 640, 480), (120, 4000, 640, 480), (30, 10, 4096, 2160),
])
def test_invalid_video_metadata(fps, frames, width, height):
    import cv2
    from visionlab.video_processing import inspect_video
    class Capture:
        def isOpened(self):
            return True
        def get(self, key):
            return {cv2.CAP_PROP_FPS: fps, cv2.CAP_PROP_FRAME_COUNT: frames,
                    cv2.CAP_PROP_FRAME_WIDTH: width, cv2.CAP_PROP_FRAME_HEIGHT: height}[key]
    with pytest.raises(ValueError):
        inspect_video(Capture())


def test_fractional_frame_rate_preserves_precision():
    import cv2
    from visionlab.video_processing import inspect_video
    class Capture:
        def isOpened(self): return True
        def get(self, key):
            return {cv2.CAP_PROP_FPS: 29.97002997002997, cv2.CAP_PROP_FRAME_COUNT: 20,
                    cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480}[key]
    assert inspect_video(Capture())['fps'] == 29.97002997002997
