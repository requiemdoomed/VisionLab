import json
import time
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from backend.main import app
from visionlab.image_processing import encode_png, decode_image


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def image():
    return encode_png(Image.new('RGB', (64, 48), '#50abcc'))


def test_health_static_and_paths(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert client.get('/api/health').json()['status'] == 'ok'
    page = client.get('/')
    assert page.status_code == 200 and 'text/html' in page.headers['content-type']
    assert '<style>' in page.text and '<script>' in page.text
    for path in ['/docs', '/openapi.json', '/api/models']:
        assert client.get(path).status_code == 200
    assert client.get('/requirements.txt').status_code == 404


def test_processed_download(client, image):
    result = client.post('/api/process', files={'file': ('image.png', image)},
                         data={'options': json.dumps({'grayscale': True})})
    assert result.status_code == 200
    assert result.headers['content-type'] == 'image/png'
    assert 'attachment' in result.headers['content-disposition']
    assert decode_image(result.content).size == (64, 48)


@pytest.mark.parametrize('options', ['not-json', '{"blur":99}', '{"brightness":-1}'])
def test_invalid_options(client, image, options):
    response = client.post('/api/process', files={'file': ('test.png', image)}, data={'options': options})
    assert response.status_code == 422
    assert response.json()['error']['message']


def test_missing_invalid_and_oversized_upload(client, image, monkeypatch):
    assert client.post('/api/process').status_code == 422
    response = client.post('/api/process', files={'file': ('fake.jpg', b'invalid')})
    assert response.status_code == 422
    assert response.json()['error']['message']
    monkeypatch.setattr('backend.main.MAX_IMAGE_BYTES', 5)
    assert client.post('/api/process', files={'file': ('test.png', image)}).status_code == 413
    assert not app.state.busy


def test_missing_model_response(client, image, monkeypatch):
    async def unavailable(*args):
        raise HTTPException(503, 'Install optional dependencies.')
    monkeypatch.setattr('backend.main.run_worker', unavailable)
    result = client.post('/api/detect', files={'file': ('image.png', image)})
    assert result.status_code == 503
    assert 'optional' in result.json()['error']['message']
    assert not app.state.busy
    assert not list(app.state.folder.iterdir())


def test_model_success_contract(client, image, monkeypatch):
    async def worker(operation, folder):
        (folder / 'output.png').write_bytes(image)
        if operation == 'remove':
            (folder / 'transparent.png').write_bytes(image)
        return {'model': 'test double', 'device': 'cpu', 'inference_ms': 1, 'count': 0, 'detections': [], 'classes': []}
    monkeypatch.setattr('backend.main.run_worker', worker)
    for path in ['/api/detect', '/api/remove-background']:
        response = client.post(path, files={'file': ('test.png', image)})
        assert response.status_code == 200
        assert response.json()['image'].startswith('data:image/png;base64,')
        if 'remove' in path:
            assert 'transparent_image' in response.json()
    assert client.get('/api/models').json()['models'][0]['status'] == 'last run succeeded'


def test_busy_and_validation(client, image):
    app.state.busy = True
    assert client.post('/api/process', files={'file': ('test.png', image)}).status_code == 429
    app.state.busy = False
    assert client.post('/api/remove-background', files={'file': ('test.png', image)}, data={'background': 'bad'}).status_code == 422
    assert client.post('/api/video', files={'file': ('test.txt', b'abc')}).status_code == 422
    assert client.get('/api/video/not-a-job').status_code == 404


def test_real_missing_dependency_worker(client, image):
    from importlib.util import find_spec
    if find_spec('ultralytics'):
        pytest.skip('Optional model package installed; do not trigger downloads in core tests.')
    result = client.post('/api/detect', files={'file': ('test.png', image)})
    assert result.status_code == 503
    assert 'requirements-models.txt' in result.json()['error']['message']


def test_real_video_round_trip(client, tmp_path):
    path = tmp_path / 'sample.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip('MJPEG encoder unavailable')
    for i in range(20):
        pixels = np.zeros((48, 64, 3), dtype=np.uint8)
        pixels[10:30, i:i+20] = (255, 150, 30)
        writer.write(pixels)
    writer.release()
    response = client.post('/api/video', files={'file': ('sample.avi', path.read_bytes())},
                           data={'options': '{"sample_every":4,"max_samples":3}'})
    assert response.status_code == 202
    job = response.json()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        status = client.get(job['status_url']).json()
        if status['state'] != 'running':
            break
        time.sleep(0.05)
    assert status['state'] == 'complete', status
    assert status['result']['processed_frames'] == 3
    assert status['progress']['completed'] == 3
    for artifact, mime in [('report', 'application/json'), ('preview', 'image/png')]:
        result = client.get(job['status_url'] + '/' + artifact)
        assert result.status_code == 200 and result.headers['content-type'].startswith(mime)
    if status['result']['export_available']:
        assert client.get(job['status_url'] + '/download').headers['content-type'] == 'video/x-msvideo'
    assert not list((app.state.folder / job['job_id']).glob('input.*'))


def test_corrupt_video_returns_failed_job(client):
    job = client.post('/api/video', files={'file': ('broken.mp4', b'not video')}).json()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        result = client.get(job['status_url']).json()
        if result['state'] != 'running':
            break
        time.sleep(.05)
    assert result['state'] == 'failed'
    assert 'decode' in result['error']


def test_worker_timeout_releases_slot(client, image, monkeypatch):
    monkeypatch.setattr('backend.main.WORKER_TIMEOUT', 0.001)
    result = client.post('/api/detect', files={'file': ('test.png', image)})
    assert result.status_code == 504
    assert not app.state.busy
    assert not list(app.state.folder.iterdir())


def test_rejected_image_does_not_mark_model_as_failed(client):
    result = client.post('/api/detect', files={'file': ('bad.png', b'not an image')})
    assert result.status_code == 422
    assert client.get('/api/models').json()['models'][0]['status'] == 'not run'


def test_detection_options_reach_worker(client, image, monkeypatch):
    async def worker(operation, folder):
        options = json.loads((folder / 'options.json').read_text())
        assert options['model'] == 'yolo11s'
        assert options['image_size'] == 960
        assert options['iou'] == .4
        (folder / 'output.png').write_bytes(image)
        return {'count': 0, 'classes': []}
    monkeypatch.setattr('backend.main.run_worker', worker)
    result = client.post('/api/detect', files={'file': ('test.png', image)},
                         data={'options': '{"model":"yolo11s","image_size":960,"iou":0.4}'})
    assert result.status_code == 200


def test_total_upload_body_limit(client, monkeypatch):
    monkeypatch.setattr('backend.main.MAX_VIDEO_BYTES', 1)
    result = client.post('/api/video', files={'file': ('test.mp4', b'x' * (1024 * 1024 + 100))})
    assert result.status_code == 413
    assert result.json()['error']['message']
    assert not app.state.busy
