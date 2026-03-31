"""
런타임에서 모델 로드 여부·플래그를 보관하기 위한 컨테이너 (향후).

지금은 싱글톤으로 쓰지 않아도 되며, 필드만 정의해 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RuntimeState:
    """추론 엔진 기동 상태 (placeholder)."""

    yolo_loaded: bool = False
    pose_loaded: bool = False
    started_at: datetime = field(default_factory=_utc_now)
