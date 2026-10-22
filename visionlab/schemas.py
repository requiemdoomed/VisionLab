from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Adjustments(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    brightness: float = Field(1, ge=0, le=2)
    contrast: float = Field(1, ge=0, le=3)
    saturation: float = Field(1, ge=0, le=3)
    sharpness: float = Field(1, ge=0, le=3)
    grayscale: bool = False
    blur: int = Field(0, ge=0, le=20)
    mode: Literal["none", "edges", "threshold"] = "none"
    threshold: int = Field(128, ge=0, le=255)


class DetectionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    confidence: float = Field(0.35, ge=0.05, le=0.95)
    model: Literal["yolo11n", "yolo11s", "demo"] = "yolo11n"
    image_size: Literal[640, 960, 1280] = 640
    iou: float = Field(0.5, ge=0.1, le=0.9)


class VideoOptions(DetectionOptions):
    sample_every: int = Field(10, ge=1, le=120)
    max_samples: int = Field(30, ge=1, le=60)
    operation: Literal["edges", "detect"] = "edges"
