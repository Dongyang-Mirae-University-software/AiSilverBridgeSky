"""카메라 상태 API."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas.camera import CameraStatusResponse

router = APIRouter(tags=["camera"])


@router.get("/camera/status", response_model=CameraStatusResponse)
def camera_status(request: Request) -> CameraStatusResponse:
    cam = getattr(request.app.state, "camera", None)
    if cam is None:
        return CameraStatusResponse(connected=False, error_message="camera service not initialized")
    s = cam.get_status()
    return CameraStatusResponse(
        connected=bool(s.get("connected")),
        active_url=s.get("active_url"),
        candidate_urls=list(s.get("candidate_urls") or []),
        reconnect_count=int(s.get("reconnect_count") or 0),
        read_fail_count=int(s.get("read_fail_count") or 0),
        last_frame_at=s.get("last_frame_at"),
        frame_width=s.get("frame_width"),
        frame_height=s.get("frame_height"),
        stale=bool(s.get("stale")),
        error_message=s.get("error_message"),
        opencv_available=bool(s.get("opencv_available", True)),
    )
