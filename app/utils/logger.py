"""
공통 로깅 설정.

시간 · 레벨 · 모듈명 · 메시지 형식을 통일한다.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_DEFAULT_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level_name: str | None = None) -> None:
    """루트 로거를 한 번만 구성한다."""
    global _configured
    if _configured:
        return
    level = logging.INFO
    if level_name:
        level = getattr(logging, level_name.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(level)
        h.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATEFMT))
        root.addHandler(h)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거. setup_logging 이후 호출되도록 main startup에서 setup_logging 선행."""
    return logging.getLogger(name)
