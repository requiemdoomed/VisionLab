"""Local resource limits shared by the API and notebook helpers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_VIDEO_BYTES = 60 * 1024 * 1024
MAX_PIXELS = 12_000_000
MAX_IMAGE_SIDE = 8192
MAX_VIDEO_PIXELS = 1920 * 1080
MAX_DURATION = 60
MAX_VIDEO_FRAMES = 3600
MAX_SAMPLES = 60
WORKER_TIMEOUT = 180
RESULT_TTL = 1800
IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}
