"""JPEG 인코딩·플레이스홀더 프레임 생성."""

from __future__ import annotations

from typing import Final

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[misc, assignment]

_LOG = __import__("logging").getLogger(__name__)

_DEFAULT_W: Final[int] = 640
_DEFAULT_H: Final[int] = 360


def ensure_bgr_uint8(frame: np.ndarray | None) -> np.ndarray | None:
    if frame is None:
        return None
    if not isinstance(frame, np.ndarray):
        return None
    if frame.size == 0:
        return None
    if frame.dtype != np.uint8:
        try:
            frame = frame.astype(np.uint8)
        except Exception:
            return None
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if cv2 is not None else None
    if len(frame.shape) == 3 and frame.shape[2] == 4 and cv2 is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def make_text_placeholder_bgr(
    text: str,
    width: int = _DEFAULT_W,
    height: int = _DEFAULT_H,
) -> np.ndarray:
    """스트림/프레임용 단색 배경 + 텍스트."""
    w, h = max(32, width), max(32, height)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if cv2 is None:
        return img
    lines = text.replace("\r", "").split("\n")[:8]
    y = 28
    for line in lines:
        cv2.putText(
            img,
            line[:120],
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return img


def encode_jpeg_bgr(
    frame_bgr: np.ndarray,
    quality: int = 85,
) -> bytes | None:
    """BGR uint8 → JPEG bytes. 실패 시 None."""
    if cv2 is None:
        return None
    q = max(1, min(100, int(quality)))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok or buf is None:
        return None
    return buf.tobytes()


def overlay_model_unavailable(bgr: np.ndarray) -> np.ndarray:
    """원본 위에 짧은 안내 텍스트."""
    if cv2 is None:
        return bgr
    out = bgr.copy()
    cv2.putText(
        out,
        "MODEL UNAVAILABLE",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return out
