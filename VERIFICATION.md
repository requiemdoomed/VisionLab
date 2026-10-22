# Verification report — VisionLab Studio 1.2.0

Checked on Linux with Python 3.12, CPU PyTorch 2.7.0, torchvision 0.22.0, Ultralytics 8.3.128, ONNX Runtime 1.21.1 and OpenCV 4.11.0. The report separates functional checks from model accuracy; one example image is not a benchmark.

## Automated tests

| Check | Result |
|---|---|
| Current model environment, integration enabled | **74 passed, 1 skipped** |
| Current model environment, ordinary test command | **70 passed, 5 skipped** |
| Frontend references and JavaScript syntax | Passed; 73 unique IDs, 38,894-byte self-contained HTML |
| Direct server launch and live HTTP requests | Passed on Linux |
| Platform launcher with fresh core environment | Passed on Linux |

The one skipped integration-enabled check exercises a missing Ultralytics installation, so it does not apply to the model environment. Four additional real model checks are opt-in in the ordinary run. The missing-package path passed in the previous core-only release verification. One upstream Starlette/AnyIO deprecation warning remains; no test failed.

Coverage includes image operations, transparency, invalid files and options, upload size limits, video metadata, fractional frame rates, subprocess timeout cleanup, correct response/download types, static-file paths, busy handling, model settings, checksum failures and black/constant-mask segmentation regressions.

## Real model inference

The test photo was the public [Ultralytics bus example](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg), 810 × 1080 pixels. It is not bundled in this project.

| Operation | Observed result |
|---|---|
| YOLO11n, 640 and 960 resolution | One bus and four people at confidence 0.35 / IoU 0.5 |
| YOLO11s, 640 and 960 resolution | One bus and four people at confidence 0.35 / IoU 0.5 |
| u2netp through ONNX Runtime | RGBA cutout retained source dimensions; alpha ranged from 0 to 255; solid compositing also worked |
| Solid black and white inputs | Background processing completed without non-finite masks or division-by-zero errors |
| YOLO video API | Two sampled frames were processed from a short clip made from the example photo; report, preview and readable AVI were returned |

The API integration suite exercised both detector choices, background removal and video detection in the real subprocess worker. `python -m visionlab.setup_models --download-only --small` also completed successfully after the optional dependencies and weights were prepared.

Observed inference time varied with warm-up and concurrent work. No speed or accuracy guarantee is inferred from these checks. Detection counts from one sample do not establish precision/recall or superiority of one model over another.

## Browser checks

A real Uvicorn server and Chromium browser completed:

- Image upload, grayscale processing, and an actual PNG download.
- YOLO11s at 960 resolution, class counts, annotated preview and actual JSON download.
- Successful model status in the About dialog.
- Neural background removal with solid compositing and an actual transparent PNG download.
- Video object detection with frame progress and an actual AVI download.
- Switching to Image Lab after entering an invalid, now-hidden Video Lab setting; processing was not blocked.
- Light/dark switching and persistence after reload, generated sample image, original/result enlargement, experimental checkpoint inference, and standalone-file offline instructions.
- Reset, workspace changes, and layouts at widths of 390, 768, 1024 and 1440 pixels.

There were no JavaScript page errors or page-wide horizontal overflow. The desktop result and phone layout were visually inspected. The downloaded PNG dimensions, transparent PNG mode and JSON contents were checked.

## Notebook and source checks

The six code cells were checked during the previous release and remain unchanged. This release adds a markdown section describing the measured training experiment. Current notebook structure was validated again.


- Notebook JSON passed `nbformat.validate`.
- All six code cells ran sequentially, including real YOLO11s detection, real ONNX segmentation and synthetic video analysis. For this check only, model execution was enabled and the local test-photo path supplied. The distributed notebook retains its portable sample path and opt-in model default.
- The regular Jupyter kernel could not start in this environment because its network-interface operation was denied. Cells were executed directly in order instead; the notebook has empty outputs and no fabricated execution counts.
- Python compilation, JavaScript syntax and shell script syntax checks passed.
- `pip check` reported no broken requirements in both the core and model environments.

## Actual training run

Five epochs of CPU fine-tuning completed on COCO8 (four training / four validation images), at 320 pixels with seed 42. The checkpoint loads through the real UI/API and produces detection results. Its mAP50–95 decreased from 0.4334706 to 0.4095754; mAP50 decreased from 0.6637403 to 0.6610307. The original detector remains the default. Full metrics, epoch history and checkpoint checksum are included in `training/demo/`.

## Remaining limits

Windows and macOS launch scripts have not been run on those operating systems. Browser testing used Chromium, not Safari or Firefox. Codec support depends on the local OpenCV build. No production-hosting audit or independent detection/segmentation benchmark was performed. The COCO8 fine-tuning validation is a tiny training smoke test with pretraining overlap, not a new accuracy benchmark.

Platform dependency selection now has automated tests, including Intel Mac pins, Windows x64 and Apple Silicon. A GitHub Actions matrix is supplied for Python 3.11/3.12 on Linux, Windows, Apple Silicon and Intel macOS, but it has not been executed remotely here.

This is a tested local project, not a claim that every possible file or machine will work without issues. The README documents input limits, model preparation and troubleshooting.

## Package

The ZIP includes complete source, tests, notebook and setup documentation. The real experimental checkpoint, its training history, measured report and upstream license are included. Standard pretrained weights download separately. Virtual environments, private files, caches and test photos are excluded. ZIP integrity and required paths were checked before delivery.

Author metadata remains **Swayam Shakya / RippedPaper**. The project has not been pushed to GitHub.
