"""헬스·상태 응답 스키마."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="ok | degraded | unavailable")
    app_name: str
    app_version: str
    uptime_seconds: float
    model_loaded: bool
    model_path: str
    model_load_error: str | None = None
    configured_classes: list[str] = Field(default_factory=list)
    model_classes: list[str] = Field(default_factory=list)
    normalized_model_classes: list[str] = Field(default_factory=list)
    missing_classes: list[str] = Field(default_factory=list)
    camera_connected: bool
    active_camera_url: str | None
    last_frame_at: str | None
    last_inference_at: str | None
    avg_inference_ms: float | None = None
    last_errors: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
