"""
향후 낙상(fall) 분석 — MediaPipe Pose 또는 키포인트 기반 분석기 연동 예정.

TODO:
- MediaPipe Pose 또는 커스텀 keypoint fall analyzer
- 연속 프레임 / 이벤트 엔진과의 스코어 정합
"""

from __future__ import annotations

from typing import Any


class PoseAnalysisService:
    """포즈·낙상 분석 (현재는 stub)."""

    def analyze(self, image: Any, detections: list[dict[str, Any]]) -> dict[str, Any]:
        """
        검출 결과와 이미지를 바탕으로 낙상 관련 특징을 반환한다.

        Returns:
            is_fall, score, person_count, reasons 등.
        """
        _ = image
        _ = detections
        return {
            "is_fall": False,
            "score": 0.12,
            "person_count": 0,
            "reasons": [],
        }
