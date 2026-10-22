# VisionLab Studio

A local computer vision workspace for editing images, detecting objects, removing backgrounds and exploring short videos.

**Author:** Swayam Shakya · [RippedPaper](https://github.com/requiemdoomed)

Open an image, adjust its details, compare the result and download it. Switch to object detection or background removal when you want to work with pretrained models. Video Lab applies edge detection or object detection to a bounded set of frames and reports the work as it finishes.

## Start here

Install **Python 3.11 or 3.12**, extract the project, and open a terminal in `VisionLab/`.

**macOS / Linux** (macOS users can also open `run_macos.command`)

```bash
bash run_mac_linux.sh
```

**Windows**

```bat
run_windows.bat
```

Open **http://127.0.0.1:8000**. Keep the terminal open; press Ctrl+C to stop.

The scripts create `.venv`, install the core dependencies and start the server on localhost. The first setup needs an internet connection. `index.html` is the application's main page, served at `/`. Opening it directly with `file://` or publishing it on GitHub Pages cannot start the Python backend.

If you are replacing an earlier copy, extract this release into a fresh folder and let the script create a new `.venv`. Do not copy the old virtual environment; the model dependencies changed.

### Enable object detection and background removal

Image adjustments and video edge detection work with the core installation. For YOLO detection and neural background removal, activate the environment and install:

```bash
# macOS / Linux
source .venv/bin/activate
# Windows Command Prompt: .venv\Scripts\activate.bat
python -m visionlab.setup_models --small
```

The setup command installs the tested CPU torch/torchvision pair, the remaining model dependencies, YOLO11n, YOLO11s and u2netp weights. Restart the server after setup. Omit `--small` if you only need the faster nano detector. If dependencies are already installed, add `--download-only`. You can also install `requirements-models.txt` manually; on Windows/Linux, installing CPU PyTorch first avoids unnecessary CUDA packages. Model dependencies include PyTorch and can be large. No paid service or GPU is required. Inference uses the CPU. An unavailable dependency or download produces an error, not a substitute result.

Standard pretrained weights are stored in `~/.cache/visionlab/`: `yolo11n.pt`, `yolo11s.pt`, and `u2netp.onnx`. The background model download is checked against its expected SHA-256 checksum before use. These caches are outside the repository. To prepare an offline machine, install dependencies and run each model once while connected, or supply trusted compatible weights in the appropriate cache. Do not load untrusted model files.

### Manual setup

```bash
python -m venv .venv
# Activate as shown above.
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Run commands from the repository root. File serving and worker imports use absolute project paths internally. Use **one Uvicorn worker**; video job state is local to that process. Do not add `--workers`.

## Training

A real five-epoch COCO8 fine-tuning run is included in `training/demo/`. The UI labels it **experimental**. Its validation mAP50–95 decreased from 0.4335 to 0.4096, so pretrained YOLO11n remains the default. Four training and four validation images verify the training path; they do not establish improved accuracy, and the original model already saw COCO during pretraining.

After model setup, reproduce the experiment from the project root:

```bash
python -m visionlab.train --demo --epochs 5 --output training/my-run
```

For your own labeled dataset:

```bash
python -m visionlab.train --data /path/to/data.yaml --epochs 30 --output training/custom-run
```

Use a local YOLO detection dataset with `train`, `val` and `names`. The trainer rejects executable download entries and overlapping image content across these splits. Keep a separate test set for evaluating a model you intend to use. A custom run writes its checkpoint and report to the selected folder; it does not silently replace the UI detector. See `training/demo/README.md` for the included experiment.

## Workspaces

| Workspace | What it does | Downloads |
|---|---|---|
| Image Lab | Brightness, contrast, saturation, sharpness, grayscale, Gaussian blur, Canny edges and thresholding | PNG |
| Object Detection | YOLO11n, YOLO11s or the experimental training demo, resolution/overlap controls, bounding boxes, labels, confidence scores and class counts | Annotated PNG and JSON |
| Background Removal | u2netp foreground segmentation through ONNX Runtime, with optional solid color | Composited PNG, transparent PNG and JSON |
| Video Lab | Metadata and sampled edge/detection frames, with actual frame-count progress | First-frame preview, JSON and MJPEG AVI when encoding succeeds |

Choose a file, set the controls and press the main action button. Results always start from the uploaded original. Reset restores the controls and clears the current result while keeping the original. Switching between the three image tools reuses your chosen image.

The About & models panel distinguishes installed packages from successful inference. It shows the last run, inference time and detector classes after a real run. Inference time excludes model loading and downloads. A package being installed does not mean its weights are available.

## Limits and behavior

- Images: PNG, JPEG or WebP; at most 12 MiB and 12 million pixels, with each side at most 8,192 pixels. Content is decoded and checked, rather than trusting MIME headers. EXIF orientation is applied. Animated inputs use their first frame; metadata is not preserved in output.
- Video: MP4, MOV, AVI or WebM containers with a locally supported codec; at most 60 MiB, 60 seconds, 3,600 frames and 1,920 pixels on the long side and 1,080 on the short side (portrait or landscape). Invalid or missing metadata is rejected. No entire video is loaded into memory.
- Sampling: every 1–120 source frames; at most 60 samples. The processor stops when the selected sample limit is reached. Results do not necessarily cover the whole clip.
- Export: sampled frames only, resized to at most 960 pixels on the longest side, with the sample-adjusted frame rate. MJPEG AVI has no audio; download and open in VLC or another compatible player. An encoder failure leaves the analysis report available.
- YOLO: pretrained COCO classes, up to 100 detections per sampled image. Small, occluded and unfamiliar objects may be missed. Confidence scores are estimates. Counts across frames are not tracked unique objects.
- Segmentation: u2netp is a compact foreground model. Hair, glass, shadows and fine edges may need manual cleanup.
- This is a single-user localhost project, without accounts or authentication. Keep the default localhost binding. Public hosting requires a deployment design with authentication, rate limits, storage isolation and reverse-proxy upload limits.
- A subprocess timeout bounds model/codec work, but is not a complete security sandbox or a hard memory quota. Use trusted local inputs.

## Notebook

```bash
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name visionlab --display-name "Python (VisionLab)"
python -m jupyter lab notebooks/VisionLab_Computer_Vision.ipynb
```

Select the VisionLab kernel. Run All creates a synthetic image, compares eight processing variants and analyzes a temporary synthetic video. The optional model cells are disabled by default; supply `assets/sample.jpg` and set `RUN_MODELS = True` to execute them. Direct notebook model calls do not have the web worker timeout.

## Troubleshooting

| Problem | What to check |
|---|---|
| Page loads but tools do not work | Use the localhost URL from the running server, not a file URL or GitHub Pages. |
| Missing module | Activate `.venv`; use `python -m pip`, not a pip from another Python installation. |
| Linux cannot create a virtual environment | Install your distribution's Python venv package. |
| OpenCV reports a missing system library | Install your distribution's OpenGL/GLib runtime libraries; server images may omit them. |
| Model unavailable | Install optional requirements, check network access and cache permissions, then retry. |
| Model download exceeds deadline | Run `python -m visionlab.setup_models --small` before opening the web app. |
| Browser will not preview a video | Its codec may still work in OpenCV. Try the analysis or convert the source to H.264 MP4. |
| Video export unavailable | Read the report; the local OpenCV build may lack a working MJPEG writer. |
| Port 8000 already in use | Stop the other process or launch Uvicorn manually with `--port 8001`. |
| Windows Python launcher unavailable | Install Python 3.11/3.12 with the launcher, or follow manual setup using `python`. |


## Licensing and dependencies

Copyright (c) 2026 Swayam Shakya (RippedPaper)
