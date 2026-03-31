"""
향후 YOLO 기반 weapon / fire 등 객체 검출.

TODO:
- Ultralytics YOLO26 로드
- yolo26n.pt 사용 (경로는 core.config.Settings.YOLO_MODEL_PATH)
- 추적·배치·비디오 파이프라인 확장
"""

from __future__ import annotations

from typing import Any


class YoloDetectionService:
    """YOLO 검출 서비스 (현재는 stub)."""

    def detect(self, image: Any) -> list[dict[str, Any]]:
        """
        이미지에 대한 검출 결과 리스트를 반환한다.

        각 원소 형식:
            {"class_name": str, "confidence": float, "bbox": [x1,y1,x2,y2]}
        """
        # stub: 실제 검출 없음 — 형식 예시는 docstring·테스트에서만 사용
        # TODO: 연동 후 list[{"class_name","confidence","bbox"}] 반환
        _ = image
        return []
