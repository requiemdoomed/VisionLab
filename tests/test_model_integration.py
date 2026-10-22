"""Opt-in integration checks for installed models and a local example photo.

VISIONLAB_MODEL_TESTS=1 VISIONLAB_SAMPLE_IMAGE=/path/photo.jpg python -m pytest -q
Prepare weights with `python -m visionlab.setup_models --small` first.
"""
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import time

import cv2
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
from backend.main import app

pytestmark = pytest.mark.skipif(os.getenv('VISIONLAB_MODEL_TESTS') != '1', reason='Real models are opt-in.')


@pytest.fixture(scope='module')
def sample_data():
    path = Path(os.environ['VISIONLAB_SAMPLE_IMAGE'])
    return path.read_bytes()


@pytest.mark.parametrize('model,size', [('yolo11n', 640), ('yolo11s', 960)])
def test_real_detector_api(sample_data, model, size):
    with TestClient(app) as client:
        result = client.post('/api/detect', files={'file': ('sample.jpg', sample_data)},
                             data={'options': json.dumps({'model': model, 'image_size': size})})
        assert result.status_code == 200, result.text[:500]
        data = result.json()
        assert data['model'].lower() == model
        assert data['count'] == len(data['detections']) == sum(data['class_counts'].values())
        assert data['inference_ms'] > 0
        assert len(data['classes']) == 80
        annotated = Image.open(BytesIO(base64.b64decode(data['image'].split(',')[1])))
        assert list(annotated.size) == data['source_size']
        for detection in data['detections']:
            assert 0 <= detection['confidence'] <= 1
            x1, y1, x2, y2 = detection['box']
            assert 0 <= x1 <= x2 <= annotated.width
            assert 0 <= y1 <= y2 <= annotated.height


def test_real_segmentation_api(sample_data):
    with TestClient(app) as client:
        response = client.post('/api/remove-background', files={'file': ('sample.jpg', sample_data)},
                               data={'background': '#172438'})
        assert response.status_code == 200, response.text[:500]
        data = response.json()
        cutout = Image.open(BytesIO(base64.b64decode(data['transparent_image'].split(',')[1])))
        composite = Image.open(BytesIO(base64.b64decode(data['image'].split(',')[1])))
        assert cutout.mode == 'RGBA'
        assert composite.getchannel('A').getextrema() == (255, 255)
        assert cutout.size == composite.size
        assert data['backend'] == 'ONNX Runtime'


def test_real_video_detection_api(sample_data, tmp_path):
    source = Image.open(BytesIO(sample_data)).convert('RGB').resize((400, 528))
    path = tmp_path / 'sample.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, source.size)
    assert writer.isOpened(), 'MJPEG encoder is needed for this integration test.'
    try:
        for _ in range(8):
            writer.write(cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    with TestClient(app) as client:
        response = client.post('/api/video', files={'file': ('sample.avi', path.read_bytes())},
                               data={'options': '{"operation":"detect","sample_every":4,"max_samples":2}'})
        assert response.status_code == 202
        url = response.json()['status_url']
        deadline = time.monotonic() + 190
        while time.monotonic() < deadline:
            job = client.get(url).json()
            if job['state'] != 'running':
                break
            time.sleep(.1)
        assert job['state'] == 'complete', job
        assert job['result']['processed_frames'] == 2
        assert all(sample['model'] == 'YOLO11n' for sample in job['result']['samples'])
        assert job['result']['export_available']
        assert client.get(url + '/download').status_code == 200
