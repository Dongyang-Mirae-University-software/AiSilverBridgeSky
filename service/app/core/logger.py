"""
공용 로거 — app.utils.logger 로 위임 (하위 호환).
"""

from __future__ import annotations

from app.utils.logger import get_logger, setup_logging

__all__ = ["get_logger", "setup_logging"]
