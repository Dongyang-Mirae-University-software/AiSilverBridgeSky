"""
바운딩 박스 후처리 (향후).

TODO: NMS, 좌표계 변환, 클램프.
"""

from __future__ import annotations


def bbox_to_dict(x1: int, y1: int, x2: int, y2: int) -> dict[str, int]:
    """[x1,y1,x2,y2]를 dict로 변환."""
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def clamp_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """이미지 경계 내로 박스를 제한한다 (시그니처만, 로직은 추후)."""
    # TODO: 실제 클램프 구현
    return x1, y1, x2, y2
