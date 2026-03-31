"""감지 API 라우트."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.detect import DetectRequest, DetectResponse
from app.services.detector_service import DetectorService

router = APIRouter(tags=["detect"])

_detector = DetectorService()


@router.post("/detect", response_model=DetectResponse)
def post_detect(body: DetectRequest) -> DetectResponse:
    """감지 파이프라인 호출 (현재 stub)."""
    return _detector.detect(body)
