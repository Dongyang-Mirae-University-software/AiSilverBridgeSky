"""
최종 이벤트 타입 결정 (fire / weapon / fall / normal).

placeholder 규칙:
- detections에 fire 또는 smoke → fire
- knife → weapon
- fall_result["is_fall"] True → fall
- 그 외 → normal
"""

from __future__ import annotations

from typing import Any


class EventEngine:
    """룰 기반 이벤트 판정 (추후 점수·히스테리시스 확장)."""

    def decide(
        self,
        detections: list[dict[str, Any]],
        fall_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Returns:
            {"type": str, "confidence": float}
        """
        for det in detections:
            name = str(det.get("class_name", "")).lower()
            conf = float(det.get("confidence", 0.0))
            if name in ("fire", "smoke"):
                return {"type": "fire", "confidence": max(conf, 0.85)}
            if name in ("knife", "weapon"):
                return {"type": "weapon", "confidence": max(conf, 0.85)}

        if fall_result.get("is_fall"):
            score = float(fall_result.get("score", 0.9))
            return {"type": "fall", "confidence": min(max(score, 0.0), 1.0)}

        return {"type": "normal", "confidence": 0.1}
