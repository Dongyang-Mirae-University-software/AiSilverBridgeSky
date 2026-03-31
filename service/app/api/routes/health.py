"""헬스·통합 상태."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request

from app.schemas.health import HealthResponse
from app.utils.class_mapper import CONFIGURED_DISPLAY_CLASSES

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """카메라/모델 일부 실패해도 200 + 상태 필드로 보고."""
    st = getattr(request.app.state, "started_at", None)
    now = time.time()
    uptime = float(now - st) if st is not None else 0.0

    cam = getattr(request.app.state, "camera", None)
    det = getattr(request.app.state, "detection", None)
    loop = getattr(request.app.state, "inference_loop", None)

    last_errors: list[str] = []
    cam_connected = False
    active_url: str | None = None
    last_frame_at: str | None = None
    summary: dict[str, Any] = {
        "configured_classes": list(CONFIGURED_DISPLAY_CLASSES),
        "model_classes": [],
        "normalized_model_classes": [],
        "missing_classes": list(CONFIGURED_DISPLAY_CLASSES),
    }

    if cam is not None:
        cs = cam.get_status()
        cam_connected = bool(cs.get("connected"))
        active_url = cs.get("active_url")
        last_frame_at = cs.get("last_frame_at")
        if cs.get("error_message"):
            last_errors.append(f"camera: {cs['error_message']}")

    model_loaded = False
    model_path = ""
    model_load_error: str | None = None
    last_inf: str | None = None
    avg_ms: float | None = None

    if det is not None:
        ds = det.get_status()
        model_loaded = bool(ds.get("model_loaded"))
        model_path = str(ds.get("model_path", ""))
        model_load_error = ds.get("load_error")
        summary = ds.get("summary") or summary
        last_inf = ds.get("last_inference_at")
        avg_ms = ds.get("avg_inference_ms_ema")
        if model_load_error:
            last_errors.append(f"model: {model_load_error}")
        le = ds.get("last_error")
        if le:
            last_errors.append(f"inference: {le}")

    if loop is not None and getattr(loop, "last_loop_error", None):
        last_errors.append(f"loop: {loop.last_loop_error}")

    status = "ok"
    if not cam_connected or not model_loaded:
        status = "degraded"
    if cam is not None and cam.get_status().get("stale"):
        status = "degraded"
    if cam is None or det is None:
        status = "unavailable"

    return HealthResponse(
        status=status,
        app_name=getattr(request.app.state, "app_name", "app"),
        app_version=getattr(request.app.state, "app_version", "0"),
        uptime_seconds=uptime,
        model_loaded=model_loaded,
        model_path=model_path,
        model_load_error=model_load_error,
        configured_classes=list(summary.get("configured_classes", CONFIGURED_DISPLAY_CLASSES)),
        model_classes=list(summary.get("model_classes", [])),
        normalized_model_classes=list(summary.get("normalized_model_classes", [])),
        missing_classes=list(summary.get("missing_classes", [])),
        camera_connected=cam_connected,
        active_camera_url=active_url,
        last_frame_at=last_frame_at,
        last_inference_at=last_inf,
        avg_inference_ms=avg_ms,
        last_errors=last_errors[:20],
        extra={},
    )
