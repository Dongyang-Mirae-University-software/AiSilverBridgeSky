"""
환경 변수 및 기본 설정.

잘못된 env 값이 있어도 앱 전체가 기동 실패하지 않도록 기본값으로 폴백한다.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_int(raw: str | None, default: int, *, min_val: int | None = None) -> int:
    if raw is None or str(raw).strip() == "":
        return default
    try:
        v = int(str(raw).strip())
        if min_val is not None and v < min_val:
            return default
        return v
    except (TypeError, ValueError):
        return default


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None or str(raw).strip() == "":
        return default
    try:
        v = float(str(raw).strip())
        if v != v:  # NaN
            return default
        return v
    except (TypeError, ValueError):
        return default


def _parse_path_list(raw: str | None, default: list[str]) -> list[str]:
    if raw is None or not str(raw).strip():
        return list(default)
    parts = [p.strip() for p in str(raw).split(",")]
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        if not p.startswith("/"):
            p = "/" + p
        out.append(p)
    return out if out else list(default)


class Settings:
    """앱 전역 설정 (환경 변수 우선, 파싱 실패 시 기본값)."""

    APP_NAME: str
    APP_VERSION: str
    HOST: str
    PORT: int
    DEBUG: bool
    MODEL_PATH: Path
    CAMERA_SCHEME: str
    CAMERA_HOST: str
    CAMERA_PORT: int
    CAMERA_PATH_CANDIDATES: list[str]
    CONF_THRESHOLD: float
    IOU_THRESHOLD: float
    FRAME_WIDTH: int
    FRAME_HEIGHT: int
    JPEG_QUALITY: int
    INFERENCE_INTERVAL_MS: int
    RECONNECT_DELAY_SECONDS: float
    FRAME_STALE_SECONDS: float
    MAX_RECONNECT_ATTEMPTS: int
    LOG_LEVEL: str
    CONSECUTIVE_READ_FAILS_BEFORE_RECONNECT: int
    DISPLAY_MODE: int
    DISPLAY_ENABLED: bool
    DISPLAY_WINDOW_NAME: str
    DISPLAY_SOURCE: str
    SCREEN_CAPTURE_REGION: tuple[int, int, int, int] | None

    def __init__(self) -> None:
        self.APP_NAME = os.getenv("APP_NAME", "SilverBridge Detection Service").strip() or "SilverBridge Detection Service"
        self.APP_VERSION = os.getenv("APP_VERSION", "0.1.0").strip() or "0.1.0"
        self.HOST = os.getenv("HOST", "0.0.0.0").strip() or "0.0.0.0"
        self.PORT = _parse_int(os.getenv("PORT"), 8000, min_val=1)
        self.DEBUG = _parse_bool(os.getenv("DEBUG"), True)

        # 하위 호환: YOLO_MODEL_PATH 도 MODEL_PATH 대안으로 허용
        _mp = os.getenv("MODEL_PATH") or os.getenv("YOLO_MODEL_PATH")
        # service/app/core/config.py → parents[3] = AiSilverBridgeSky
        _default_model = (
            Path(__file__).resolve().parents[3]
            / "training"
            / "runs"
            / "fire_smoke_knife_yolo26"
            / "weights"
            / "best.pt"
        )
        if _mp and str(_mp).strip():
            self.MODEL_PATH = Path(str(_mp).strip()).expanduser()
        else:
            self.MODEL_PATH = _default_model

        self.CAMERA_SCHEME = (os.getenv("CAMERA_SCHEME") or "http").strip().lower() or "http"
        self.CAMERA_HOST = (os.getenv("CAMERA_HOST") or "127.0.0.1").strip() or "127.0.0.1"
        self.CAMERA_PORT = _parse_int(os.getenv("CAMERA_PORT"), 4747, min_val=1)
        self.CAMERA_PATH_CANDIDATES = _parse_path_list(
            os.getenv("CAMERA_PATH_CANDIDATES"),
            ["/video", "/mjpegfeed", "/"],
        )

        self.CONF_THRESHOLD = _parse_float(os.getenv("CONF_THRESHOLD"), 0.35)
        self.IOU_THRESHOLD = _parse_float(os.getenv("IOU_THRESHOLD"), 0.45)
        self.FRAME_WIDTH = _parse_int(os.getenv("FRAME_WIDTH"), 1280, min_val=1)
        self.FRAME_HEIGHT = _parse_int(os.getenv("FRAME_HEIGHT"), 720, min_val=1)
        self.JPEG_QUALITY = _parse_int(os.getenv("JPEG_QUALITY"), 85, min_val=1)
        if self.JPEG_QUALITY > 100:
            self.JPEG_QUALITY = 100

        self.INFERENCE_INTERVAL_MS = _parse_int(os.getenv("INFERENCE_INTERVAL_MS"), 150, min_val=1)
        self.RECONNECT_DELAY_SECONDS = _parse_float(os.getenv("RECONNECT_DELAY_SECONDS"), 3.0)
        self.FRAME_STALE_SECONDS = _parse_float(os.getenv("FRAME_STALE_SECONDS"), 5.0)
        self.MAX_RECONNECT_ATTEMPTS = _parse_int(os.getenv("MAX_RECONNECT_ATTEMPTS"), 0, min_val=0)
        self.LOG_LEVEL = (os.getenv("LOG_LEVEL") or "INFO").strip().upper() or "INFO"
        self.CONSECUTIVE_READ_FAILS_BEFORE_RECONNECT = _parse_int(
            os.getenv("CONSECUTIVE_READ_FAILS_BEFORE_RECONNECT"),
            15,
            min_val=1,
        )
        mode = _parse_int(os.getenv("DISPLAY_MODE"), 0, min_val=0)
        if mode not in (0, 1, 2):
            mode = 0
        self.DISPLAY_MODE = mode
        # 하위 호환: 기존 DISPLAY_ENABLED=true 면 mode=1로 취급
        legacy_enabled = _parse_bool(os.getenv("DISPLAY_ENABLED"), False)
        if legacy_enabled and self.DISPLAY_MODE == 0:
            self.DISPLAY_MODE = 1
        self.DISPLAY_ENABLED = _parse_bool(os.getenv("DISPLAY_ENABLED"), False)
        self.DISPLAY_WINDOW_NAME = (os.getenv("DISPLAY_WINDOW_NAME") or "SilverBridge Live").strip() or "SilverBridge Live"
        src = (os.getenv("DISPLAY_SOURCE") or "annotated").strip().lower()
        if src not in ("annotated", "raw", "auto"):
            src = "annotated"
        self.DISPLAY_SOURCE = src
        self.SCREEN_CAPTURE_REGION = None
        region_raw = os.getenv("SCREEN_CAPTURE_REGION", "").strip()
        if region_raw:
            parts = [p.strip() for p in region_raw.split(",")]
            if len(parts) == 4:
                try:
                    x, y, w, h = [int(v) for v in parts]
                    if w > 0 and h > 0:
                        self.SCREEN_CAPTURE_REGION = (x, y, w, h)
                except ValueError:
                    self.SCREEN_CAPTURE_REGION = None

    def build_camera_urls(self) -> list[str]:
        """설정 기반 DroidCam 등 HTTP 스트림 URL 후보."""
        base = f"{self.CAMERA_SCHEME}://{self.CAMERA_HOST}:{self.CAMERA_PORT}"
        urls: list[str] = []
        for path in self.CAMERA_PATH_CANDIDATES:
            if path == "/":
                urls.append(f"{base}/")
            else:
                urls.append(f"{base}{path}")
        return urls


@lru_cache
def get_settings() -> Settings:
    """앱 전역에서 재사용하는 설정 인스턴스."""
    return Settings()


def reset_settings_cache() -> None:
    """테스트 등에서 설정 캐시를 비울 때 사용."""
    get_settings.cache_clear()
