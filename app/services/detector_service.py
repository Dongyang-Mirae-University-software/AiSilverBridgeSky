"""
/detect 오케스트레이션.

실제 추론 대신 서비스 호출 순서만 고정한다:
decode → yolo → pose → event_engine → DetectResponse
"""

from __future__ import annotations

from app.core.config import get_settings
from app.schemas.detect import (
    DetectRequest,
    DetectResponse,
    DetectionItem,
    FallAnalysis,
)
from app.services.event_engine import EventEngine
from app.services.image_decoder import decode_base64_image
from app.services.pose_service import PoseAnalysisService
from app.services.yolo_service import YoloDetectionService
from app.utils.timing import now_utc_iso


class DetectorService:
    """통합 감지 파이프라인 (stub 단계)."""

    def __init__(self) -> None:
        self._yolo = YoloDetectionService()
        self._pose = PoseAnalysisService()
        self._events = EventEngine()
        self._settings = get_settings()

    def detect(self, request: DetectRequest) -> DetectResponse:
        """요청을 받아 고정 스키마의 stub 응답을 생성한다."""
        decoded_payload = decode_base64_image(request.image)

        raw_detections = self._yolo.detect(decoded_payload)

        fall_raw = self._pose.analyze(decoded_payload, raw_detections)
        decision = self._events.decide(raw_detections, fall_raw)

        detections_models = [
            DetectionItem(
                class_name=d["class_name"],
                confidence=float(d["confidence"]),
                bbox=list(d["bbox"]),
            )
            for d in raw_detections
        ]

        fall_analysis = FallAnalysis(
            is_fall=bool(fall_raw.get("is_fall", False)),
            person_count=int(fall_raw.get("person_count", 0)),
            reasons=list(fall_raw.get("reasons", [])),
        )

        return DetectResponse(
            type=decision["type"],
            confidence=float(decision["confidence"]),
            timestamp=now_utc_iso(),
            detections=detections_models,
            fall_analysis=fall_analysis,
            meta={
                "mode": "stub",
                "yolo_model": "not_loaded",
                "pose_model": "not_loaded",
                "app_version": self._settings.APP_VERSION,
            },
        )
