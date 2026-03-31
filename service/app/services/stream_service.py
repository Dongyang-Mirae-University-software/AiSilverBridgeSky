"""MJPEG multipart 스트림 생성."""

from __future__ import annotations

import time
from typing import Generator

import numpy as np

from app.core.config import Settings
from app.services.camera_service import CameraService
from app.services.detection_service import DetectionService
from app.utils.image_utils import encode_jpeg_bgr, make_text_placeholder_bgr
from app.utils.logger import get_logger

_LOG = get_logger(__name__)

_BOUNDARY = b"frame"


class StreamService:
    def __init__(
        self,
        settings: Settings,
        camera: CameraService,
        detection: DetectionService,
    ) -> None:
        self._settings = settings
        self._camera = camera
        self._detection = detection

    def get_jpeg_for_frame_endpoint(self) -> tuple[bytes, str]:
        """
        /frame 단일 JPEG.
        returns (bytes, content_type) — 실패 시 placeholder.
        """
        raw = self._camera.get_latest_frame_copy()
        dets, ann, _meta = self._detection.get_latest_for_api()
        frame = ann if ann is not None else raw
        if frame is None:
            ph = make_text_placeholder_bgr("NO FRAME\n(camera disconnected)", self._settings.FRAME_WIDTH, self._settings.FRAME_HEIGHT)
            jpeg = encode_jpeg_bgr(ph, self._settings.JPEG_QUALITY)
            return jpeg or b"", "image/jpeg"
        jpeg = encode_jpeg_bgr(frame, self._settings.JPEG_QUALITY)
        if not jpeg:
            ph = make_text_placeholder_bgr("JPEG ENCODE ERROR", self._settings.FRAME_WIDTH, self._settings.FRAME_HEIGHT)
            jpeg = encode_jpeg_bgr(ph, self._settings.JPEG_QUALITY) or b""
        return jpeg, "image/jpeg"

    def mjpeg_generator(self) -> Generator[bytes, None, None]:
        """동기 제너레이터 — StreamingResponse 에 전달."""
        while True:
            try:
                raw = self._camera.get_latest_frame_copy()
                _, ann, _ = self._detection.get_latest_for_api()
                frame: np.ndarray | None = ann if ann is not None else raw
                if frame is None:
                    frame = make_text_placeholder_bgr(
                        "NO FRAME",
                        self._settings.FRAME_WIDTH,
                        self._settings.FRAME_HEIGHT,
                    )
                jpeg = encode_jpeg_bgr(frame, self._settings.JPEG_QUALITY)
                if not jpeg:
                    time.sleep(0.1)
                    continue
                yield (
                    b"--" + _BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            except Exception as e:  # noqa: BLE001
                _LOG.exception("MJPEG chunk 오류: %s", e)
                time.sleep(0.2)
            time.sleep(max(0.01, self._settings.INFERENCE_INTERVAL_MS / 1000.0 / 2))


def build_streaming_response_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
