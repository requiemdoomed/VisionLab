"""Pretrained YOLO detection with explicit CPU and quality controls."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
from time import perf_counter

from PIL import Image
from .schemas import DetectionOptions
from .config import ROOT

MODEL_NAMES = {"yolo11n": "YOLO11n", "yolo11s": "YOLO11s", "demo": "COCO8 fine-tuning demo"}


class ModelUnavailable(RuntimeError):
    """An optional dependency or trusted model weight could not be loaded."""


@lru_cache(maxsize=3)
def load_detector(model_name: str = "yolo11n"):
    # Only these bundled model choices are accepted; never accept arbitrary paths.
    if model_name not in MODEL_NAMES:
        raise ValueError("Choose a supported detector.")
    try:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(2)
        cache = Path.home() / ".cache" / "visionlab"
        cache.mkdir(parents=True, exist_ok=True)
        if model_name == "demo":
            path = ROOT / "training" / "demo" / "best.pt"
            if not path.is_file():
                raise FileNotFoundError("The demo checkpoint is missing from training/demo/.")
        else:
            path = cache / f"{model_name}.pt"
        return YOLO(str(path))
    except Exception as exc:
        if model_name == "demo":
            raise ModelUnavailable(
                "The experimental checkpoint could not load. Install the optional model "
                "dependencies and restore training/demo/best.pt from the project download."
            ) from exc
        raise ModelUnavailable(
            f"{MODEL_NAMES[model_name]} could not load. Run python -m visionlab.setup_models "
            f"to install requirements-models.txt and prepare weights, or place trusted "
            f"{model_name}.pt weights in ~/.cache/visionlab/."
        ) from exc


def detect_objects(image: Image.Image, confidence: float = 0.35, *,
                   model: str = "yolo11n", image_size: int = 640, iou: float = 0.5):
    options = DetectionOptions(confidence=confidence, model=model, image_size=image_size, iou=iou)
    detector = load_detector(options.model)
    # Composite transparent input so invisible RGB values do not become objects.
    rgba = image.convert("RGBA")
    source = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    start = perf_counter()
    prediction = detector.predict(source, conf=options.confidence, iou=options.iou,
                                  device="cpu", imgsz=options.image_size, max_det=100,
                                  verbose=False, save=False)[0]
    elapsed = round((perf_counter() - start) * 1000, 1)
    detections = []
    for box in prediction.boxes:
        class_id = int(box.cls.item())
        detections.append({"class_id": class_id, "label": prediction.names[class_id],
                           "confidence": round(float(box.conf.item()), 4),
                           "box": [round(float(v), 1) for v in box.xyxy[0].tolist()]})
    detections.sort(key=lambda item: item["confidence"], reverse=True)
    annotated = Image.fromarray(prediction.plot()[:, :, ::-1])
    return annotated, {"model": MODEL_NAMES[options.model], "device": "cpu", "inference_ms": elapsed,
                       "image_size": options.image_size, "confidence_threshold": options.confidence,
                       "iou_threshold": options.iou, "source_size": list(image.size),
                       "count": len(detections), "detections": detections,
                       "class_counts": dict(sorted(Counter(d["label"] for d in detections).items())),
                       "classes": list(prediction.names.values())}
