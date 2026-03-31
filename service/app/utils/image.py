"""
이미지 전처리·변환 공용 함수 (향후).

TODO: OpenCV/NumPy 배열 변환, 리사이즈, 정규화 등.
"""

from __future__ import annotations

from typing import Any


def placeholder_image_info(image_payload: Any) -> dict[str, Any]:
    """stub 단계: 입력 페이로드 메타만 반환."""
    return {
        "has_payload": image_payload is not None,
        "kind": type(image_payload).__name__,
    }
