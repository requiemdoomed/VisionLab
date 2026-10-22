"""Same-origin local API. Run with: python -m uvicorn backend.main:app"""
import asyncio
import base64
from contextlib import asynccontextmanager
from importlib.util import find_spec
import json
import logging
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from visionlab.config import (ROOT, MAX_IMAGE_BYTES, MAX_VIDEO_BYTES, VIDEO_EXTENSIONS,
                              WORKER_TIMEOUT, RESULT_TTL)
from visionlab.image_processing import decode_image, encode_png, process_image
from visionlab.schemas import Adjustments, DetectionOptions, VideoOptions

logger = logging.getLogger("visionlab")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.folder = Path(tempfile.mkdtemp(prefix="visionlab-"))
    app.state.jobs = {}
    app.state.busy = False
    app.state.tasks = set()
    app.state.models = {}

    async def reap():
        while True:
            await asyncio.sleep(30)
            for job_id, job in list(app.state.jobs.items()):
                if job["state"] != "running" and time.monotonic() - job.get("finished", job["created"]) > RESULT_TTL:
                    shutil.rmtree(app.state.folder / job_id, ignore_errors=True)
                    del app.state.jobs[job_id]

    cleaner = asyncio.create_task(reap())
    try:
        yield
    finally:
        cleaner.cancel()
        for task in list(app.state.tasks):
            task.cancel()
        await asyncio.gather(cleaner, *app.state.tasks, return_exceptions=True)
        shutil.rmtree(app.state.folder, ignore_errors=True)


app = FastAPI(title="VisionLab Studio", version="1.2.0", lifespan=lifespan)


class UploadLimit:
    """Enforce total body size even when the client omits Content-Length."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        used = 0
        async def limited_receive():
            nonlocal used
            message = await receive()
            used += len(message.get("body", b""))
            if used > MAX_VIDEO_BYTES + 1024 * 1024:
                raise HTTPException(413, "Upload exceeds the 60 MiB request limit.")
            return message
        await self.app(scope, limited_receive, send)


app.add_middleware(UploadLimit)


@app.exception_handler(StarletteHTTPException)
async def http_error(request, exc):
    return JSONResponse({"error": {"message": str(exc.detail)}}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    return JSONResponse({"error": {"message": "Invalid or missing request parameters.",
                                  "fields": [{"field": ".".join(map(str, e["loc"])),
                                              "message": e["msg"]} for e in exc.errors()]}}, status_code=422)


@app.exception_handler(Exception)
async def unexpected_error(request, exc):
    logger.exception("Request failed: %s", request.url.path)
    return JSONResponse({"error": {"message": "Unexpected server error. Check the server terminal."}}, status_code=500)


def parse_options(model, text):
    try:
        return model.model_validate_json(text)
    except ValueError as exc:
        raise HTTPException(422, "Invalid options: use the documented parameter ranges.") from exc


async def image_upload(file: UploadFile):
    try:
        data = await file.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Images must be smaller than 12 MiB.")
        try:
            image = await run_in_threadpool(decode_image, data)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return data, image
    finally:
        await file.close()


def reserve():
    # All reservations happen on the event loop, before the next await.
    if app.state.busy:
        raise HTTPException(429, "The studio is processing another request. Try again when it finishes.")
    app.state.busy = True


async def run_worker(operation: str, folder: Path):
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "backend.worker", operation, str(folder), cwd=ROOT,
        stdout=asyncio.subprocess.DEVNULL)
    try:
        await asyncio.wait_for(process.wait(), WORKER_TIMEOUT)
    except (TimeoutError, asyncio.CancelledError):
        if process.returncode is None:
            process.kill()
        await process.wait()
        if asyncio.current_task().cancelling():
            raise
        raise HTTPException(504, "Processing exceeded 180 seconds. Try a smaller input; model downloads may need to be completed first.")
    error_path = folder / "error.json"
    if error_path.exists():
        error = json.loads(error_path.read_text())
        raise HTTPException(error["status"], error["message"])
    if process.returncode != 0 or not (folder / "result.json").exists():
        raise HTTPException(500, "Processing stopped unexpectedly. Check available memory and the server terminal.")
    return json.loads((folder / "result.json").read_text())


def image_payload(path: Path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(ROOT / "index.html", media_type="text/html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.2.0", "author": "Swayam Shakya", "busy": app.state.busy}


@app.get("/api/models")
async def models():
    return {"device": "cpu", "lifecycle": "Models load inside each isolated request; reused across video frames.",
            "models": [{"id": key, "name": name, "installed": find_spec(package) is not None,
                        "status": app.state.models.get(key, {}).get("status", "not run"),
                        "last_run": app.state.models.get(key),
                        "classes": app.state.models.get(key, {}).get("classes", []),
                        "note": note}
                       for key, name, package, note in [
                           ("detect", "YOLO11n / YOLO11s / training demo", "ultralytics", "80 COCO classes. The training demo is experimental; pretrained YOLO11n is the default."),
                           ("remove", "U²-Net small (u2netp)", "onnxruntime", "Foreground segmentation; hair, glass and fine edges can be imperfect.")]]}


@app.post("/api/process")
async def process(file: UploadFile = File(...), options: str = Form("{}")):
    settings = parse_options(Adjustments, options)
    reserve()
    try:
        _, image = await image_upload(file)
        output = await run_in_threadpool(lambda: encode_png(process_image(image, settings)))
        return Response(output, media_type="image/png",
                        headers={"Content-Disposition": 'attachment; filename="visionlab-processed.png"'})
    finally:
        app.state.busy = False


async def model_image(operation, file, options):
    reserve()
    folder = None
    worker_started = False
    try:
        data, _ = await image_upload(file)
        folder = Path(tempfile.mkdtemp(dir=app.state.folder))
        (folder / "input").write_bytes(data)
        (folder / "options.json").write_text(json.dumps(options))
        worker_started = True
        result = await run_worker(operation, folder)
        app.state.models[operation] = {"status": "last run succeeded", **result}
        response = {"image": image_payload(folder / "output.png"), **result}
        if operation == "remove":
            response["transparent_image"] = image_payload(folder / "transparent.png")
        return response
    except HTTPException as exc:
        if worker_started:
            app.state.models[operation] = {"status": "last run failed", "message": str(exc.detail)}
        raise
    finally:
        if folder:
            shutil.rmtree(folder, ignore_errors=True)
        app.state.busy = False


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), options: str = Form("{}")):
    settings = parse_options(DetectionOptions, options)
    return await model_image("detect", file, settings.model_dump())


@app.post("/api/remove-background")
async def remove(file: UploadFile = File(...), background: str = Form("")):
    if background and not re.fullmatch(r"#[0-9a-fA-F]{6}", background):
        raise HTTPException(422, "Use a six-digit hex background color.")
    return await model_image("remove", file, {"background": background or None})


async def finish_video(job_id, folder):
    job = app.state.jobs[job_id]
    try:
        job["result"] = await run_worker("video", folder)
        job["state"] = "complete"
        samples = job["result"]["samples"]
        if job["result"]["operation"] == "detect" and samples:
            app.state.models["detect"] = {"status": "last run succeeded", **samples[-1]}
    except HTTPException as exc:
        job.update(state="failed", error=str(exc.detail))
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Video task failed")
        job.update(state="failed", error="Video processing failed. Check the terminal.")
    finally:
        job["finished"] = time.monotonic()
        for source in folder.glob("input.*"):
            source.unlink(missing_ok=True)
        app.state.busy = False


@app.post("/api/video", status_code=202)
async def video(file: UploadFile = File(...), options: str = Form("{}")):
    settings = parse_options(VideoOptions, options)
    extension = Path(file.filename or "").suffix.lower()
    if extension not in VIDEO_EXTENSIONS:
        raise HTTPException(422, "Use an MP4, MOV, AVI, or WebM video.")
    reserve()
    folder = None
    started = False
    try:
        if len(app.state.jobs) >= 8:
            completed = [key for key, value in app.state.jobs.items() if value["state"] != "running"]
            if completed:
                oldest = completed[0]
                shutil.rmtree(app.state.folder / oldest, ignore_errors=True)
                del app.state.jobs[oldest]
        job_id = uuid.uuid4().hex
        folder = app.state.folder / job_id
        folder.mkdir()
        total = 0
        source = "input" + extension
        with (folder / source).open("wb") as stream:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_VIDEO_BYTES:
                    raise HTTPException(413, "Videos must be smaller than 60 MiB.")
                stream.write(chunk)
        if not total:
            raise HTTPException(422, "The video file is empty.")
        (folder / "options.json").write_text(json.dumps({"source": source, **settings.model_dump()}))
        app.state.jobs[job_id] = {"state": "running", "created": time.monotonic()}
        task = asyncio.create_task(finish_video(job_id, folder))
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        started = True
        return {"job_id": job_id, "status_url": f"/api/video/{job_id}"}
    finally:
        await file.close()
        if not started:
            app.state.busy = False
            if folder:
                shutil.rmtree(folder, ignore_errors=True)


def get_job(job_id):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id) or job_id not in app.state.jobs:
        raise HTTPException(404, "Video result not found or expired. Upload the video again.")
    return app.state.jobs[job_id], app.state.folder / job_id


@app.get("/api/video/{job_id}")
async def video_status(job_id: str):
    job, folder = get_job(job_id)
    progress = {"completed": 0, "total": None}
    if (folder / "progress.json").exists():
        progress = json.loads((folder / "progress.json").read_text())
    return {key: value for key, value in job.items() if key not in {"created", "finished"}} | {"progress": progress}


@app.get("/api/video/{job_id}/{artifact}")
async def video_artifact(job_id: str, artifact: str):
    job, folder = get_job(job_id)
    allowed = {"download": ("output.avi", "video/x-msvideo"),
               "preview": ("output.png", "image/png"), "report": ("result.json", "application/json")}
    if artifact not in allowed or job["state"] != "complete":
        raise HTTPException(404, "This result is not available.")
    if artifact == "download" and not job["result"]["export_available"]:
        raise HTTPException(409, "Video encoding was unavailable. Download the frame report instead.")
    name, mime = allowed[artifact]
    return FileResponse(folder / name, media_type=mime, filename="visionlab-" + name)
