# Changes

## 1.2.0

- Made index.html self-contained, with responsive tabs, persistent light/dark themes, enlarged previews, a sample image and offline startup help.
- Added a real five-epoch fine-tuning demo, portable checkpoint, measured validation results and an experimental model choice. Kept the pretrained detector as the default because the demo did not improve validation accuracy.
- Added local-dataset validation and repeatable training commands.
- Corrected Intel Mac dependency versions and Python selection in launchers; added a macOS double-click launcher and Windows Python fallback.
- Added platform tests, frontend checks and a GitHub Actions matrix for Linux, Windows and both Mac architectures. Native Windows/macOS results are still pending.

## 1.1.0

- Added YOLO11s alongside YOLO11n, three inference resolutions, an overlap threshold, and per-class counts. Detection results are sorted by confidence.
- Verified both detectors with real pretrained weights and a public example image.
- Added `python -m visionlab.setup_models --small` to install the CPU model stack and prepare weights before launching the studio.
- Replaced the rembg wrapper with direct U²-Net inference through ONNX Runtime. This avoids installing two packages that both provide `cv2`. The model download has a fixed SHA-256 checksum and uses an atomic file replacement.
- Fixed black-image normalization and constant-mask handling. Background removal preserves existing transparency; detection does not use invisible RGB pixels.
- Preserved fractional video frame rates, bounded portrait exports by their longest side, and added a warning when decoding ends before the planned sample count.
- Tightened image/video dimension checks, made hidden controls inactive for form validation, added request timeouts, and kept rejected uploads from being reported as model failures.
- Added model adapter regressions and opt-in API tests for real detector, segmentation and video inference. Updated setup, notebook and verification notes.

## 1.0.0

Initial local image/video studio with a FastAPI backend, reusable processing modules and companion notebook.
