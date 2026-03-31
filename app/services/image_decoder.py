"""
입력 이미지 디코딩 (향후 base64 / URL / multipart).

지금은 경계만 분리하고 실제 디코딩은 하지 않는다.
"""

from __future__ import annotations

from typing import Any


def decode_base64_image(image_str: str | None) -> dict[str, Any]:
    """
    base64 또는 원시 문자열을 이미지로 변환하기 위한 진입점.

    Returns:
        decoded 여부와 원시 입력 일부를 담은 dict.
        TODO: 성공 시 numpy.ndarray 또는 bytes 반환 구조로 확장.
    """
    if image_str is None or image_str.strip() == "":
        return {"decoded": False, "raw": None, "error": "empty_input"}
    return {"decoded": False, "raw": image_str}
