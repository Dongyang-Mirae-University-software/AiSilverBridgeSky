"""실시간 감지 API 스키마 (/classes, /detections/current)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassesResponse(BaseModel):
    configured_classes: list[str] = Field(default_factory=list)
    model_classes: list[str] = Field(default_factory=list)
    normalized_model_classes: list[str] = Field(default_factory=list)
    missing_classes: list[str] = Field(default_factory=list)


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionItemOut(BaseModel):
    class_id: int
    original_class_name: str
    display_class_name: str
    confidence: float
    bbox: BBox
    detected_at: str | None = None


class DetectionCurrentResponse(BaseModel):
    detected_at: str | None = None
    inference_ms: float | None = None
    total_detections: int = 0
    detections: list[DetectionItemOut] = Field(default_factory=list)
    model_loaded: bool = False
    frame_available: bool = False
    last_error: str | None = None
