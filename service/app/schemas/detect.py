"""
/detect 요청·응답 스키마.

응답 형식은 실제 YOLO/Pose 연동 후에도 유지하는 것을 전제로 고정한다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    """클라이언트 입력 (이미지는 base64 등 문자열로 확장 예정)."""

    image: str | None = None


class DetectionItem(BaseModel):
    """단일 검출 박스."""

    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[int] = Field(
        default_factory=list,
        description="[x1, y1, x2, y2] 형식 예정",
    )


class FallAnalysis(BaseModel):
    """낙상 분석 결과."""

    is_fall: bool = False
    person_count: int = 0
    reasons: list[str] = Field(default_factory=list)


class DetectResponse(BaseModel):
    """통합 감지 응답."""

    type: str
    confidence: float
    timestamp: str
    detections: list[DetectionItem] = Field(default_factory=list)
    fall_analysis: FallAnalysis | None = None
    meta: dict = Field(default_factory=dict)
