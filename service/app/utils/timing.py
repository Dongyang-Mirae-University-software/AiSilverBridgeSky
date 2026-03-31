"""시간·지연 측정 유틸."""

from __future__ import annotations

from datetime import datetime, timezone


def now_utc_iso() -> str:
    """현재 시각을 UTC ISO8601 문자열로 반환한다 (초 단위, Z 접미사)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# TODO: 추후 context manager 로 구간 측정 (예: with measure_ms() as m)
