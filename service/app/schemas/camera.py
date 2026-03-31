"""카메라 상태 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CameraStatusResponse(BaseModel):
    connected: bool
    active_url: str | None = None
    candidate_urls: list[str] = Field(default_factory=list)
    reconnect_count: int = 0
    read_fail_count: int = 0
    last_frame_at: str | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    stale: bool = False
    error_message: str | None = None
    opencv_available: bool = True
