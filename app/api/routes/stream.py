"""단일 프레임·MJPEG 스트림."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from app.services.stream_service import build_streaming_response_headers

router = APIRouter(tags=["stream"])


@router.get("/frame")
def get_frame(request: Request) -> Response:
    svc = getattr(request.app.state, "stream", None)
    if svc is None:
        return Response(content=b"", media_type="image/jpeg", status_code=503)
    jpeg, ctype = svc.get_jpeg_for_frame_endpoint()
    return Response(content=jpeg, media_type=ctype)


@router.get("/stream")
def get_stream(request: Request) -> StreamingResponse:
    svc = getattr(request.app.state, "stream", None)
    if svc is None:
        return StreamingResponse(iter(()), media_type="text/plain", status_code=503)

    gen = svc.mjpeg_generator()
    headers = build_streaming_response_headers()
    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )
