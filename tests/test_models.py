"""Offline regressions for model adapters; no weights are downloaded here."""
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError
from visionlab.schemas import DetectionOptions
from visionlab import detection, segmentation


@pytest.mark.parametrize('options', [dict(model='../custom'), dict(image_size=641), dict(iou=0),
                                    dict(iou=float('nan')), dict(model='yolo26n')])
def test_detector_choices_are_bounded(options):
    with pytest.raises(ValidationError):
        DetectionOptions(**options)


def test_detection_passes_options_and_hides_invisible_rgb(monkeypatch):
    captured = {}
    class Detector:
        def predict(self, source, **options):
            captured.update(options)
            assert source.getpixel((0, 0)) == (255, 255, 255)
            return [SimpleNamespace(boxes=[], names={0: 'person'},
                                    plot=lambda: np.zeros((3, 4, 3), dtype=np.uint8))]
    def load(name):
        assert name == 'yolo11s'
        return Detector()
    monkeypatch.setattr(detection, 'load_detector', load)
    image, result = detection.detect_objects(Image.new('RGBA', (4, 3), (255, 0, 0, 0)),
                                             model='yolo11s', image_size=960, iou=.4)
    assert captured['imgsz'] == 960 and captured['iou'] == .4
    assert captured['device'] == 'cpu'
    assert result['model'] == 'YOLO11s'
    assert result['class_counts'] == {} and result['count'] == 0
    assert image.size == (4, 3)


class ConstantMaskSession:
    def __init__(self, value):
        self.value = value
    def get_inputs(self):
        return [SimpleNamespace(name='input')]
    def run(self, _, inputs):
        tensor = inputs['input']
        assert tensor.shape == (1, 3, 320, 320)
        assert tensor.dtype == np.float32 and np.isfinite(tensor).all()
        return [np.full((1, 1, 320, 320), self.value, dtype=np.float32)]


@pytest.mark.parametrize('value,expected', [(0, 0), (1, 255), (.5, 127)])
def test_black_image_and_constant_mask_are_finite(value, expected):
    mask = segmentation.foreground_mask(Image.new('RGB', (16, 12), 'black'), ConstantMaskSession(value))
    assert mask.size == (16, 12)
    assert np.all(np.asarray(mask) == expected)


def test_nonfinite_mask_rejected():
    with pytest.raises(RuntimeError, match='non-finite'):
        segmentation.foreground_mask(Image.new('RGB', (16, 12)), ConstantMaskSession(float('nan')))


def test_segmentation_preserves_source_alpha_and_composites(monkeypatch):
    monkeypatch.setattr(segmentation, 'load_session', lambda: ConstantMaskSession(1))
    source = Image.new('RGBA', (16, 12), (230, 20, 10, 0))
    composite, cutout, report = segmentation.remove_background(source, '#123456')
    assert cutout.getchannel('A').getextrema() == (0, 0)
    assert composite.getpixel((0, 0)) == (18, 52, 86, 255)
    assert report['backend'] == 'ONNX Runtime'


def test_bad_download_is_not_cached(tmp_path, monkeypatch):
    from io import BytesIO
    destination = tmp_path / 'u2netp.onnx'
    monkeypatch.setattr(segmentation, 'model_path', lambda: destination)
    monkeypatch.setattr(segmentation.urllib.request, 'urlopen', lambda *a, **kw: BytesIO(b'bad weights'))
    with pytest.raises(ValueError, match='checksum'):
        segmentation.prepare_weights()
    assert not destination.exists()
    assert not list(tmp_path.glob('*.download'))
