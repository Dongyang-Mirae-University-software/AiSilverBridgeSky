"""
DroidCam 등 HTTP MJPEG 스트림 수신, 최신 프레임 캐시, 자동 재연결.

메인 스레드를 블로킹하지 않는다.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.core.config import Settings
from app.utils.logger import get_logger

_LOG = get_logger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[misc, assignment]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CameraService:
    """백그라운드에서 프레임을 읽고 최신 프레임만 유지한다."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._cap: Any = None
        self._active_url: str | None = None
        self._connected = False
        self._latest_frame: np.ndarray | None = None
        self._last_frame_at: datetime | None = None
        self._frame_size: tuple[int, int] | None = None

        self._reconnect_count = 0
        self._read_fail_count = 0
        self._read_fail_streak = 0
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="camera-read", daemon=True)
        self._thread.start()
        _LOG.info("카메라 백그라운드 스레드 시작")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._release_cap_safe()
        _LOG.info("카메라 서비스 중지 요청 완료")

    def _release_cap_safe(self) -> None:
        if self._cap is None:
            return
        try:
            self._cap.release()
        except Exception as e:  # noqa: BLE001
            _LOG.debug("VideoCapture release 경고: %s", e)
        finally:
            self._cap = None
            self._connected = False
            self._active_url = None

    def _try_open_url(self, url: str) -> bool:
        if cv2 is None:
            self._last_error = "opencv 미설치"
            return False
        cap = None
        try:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        except Exception as e:  # noqa: BLE001
            self._last_error = f"VideoCapture 생성 실패: {e}"
            _LOG.warning("%s (%s)", self._last_error, url)
            return False

        if cap is None or not cap.isOpened():
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            self._last_error = "VideoCapture open 실패"
            return False

        # 실제 프레임 1장 확보까지 성공해야 연결로 인정
        for _ in range(30):
            ok, frame = cap.read()
            if ok and frame is not None and getattr(frame, "size", 0) > 0:
                self._cap = cap
                self._active_url = url
                self._connected = True
                self._last_error = None
                self._store_frame(frame)
                _LOG.info("카메라 연결 성공: %s", url)
                return True
            time.sleep(0.05)

        try:
            cap.release()
        except Exception:
            pass
        self._last_error = "첫 프레임 수신 실패"
        return False

    def _connect_any(self) -> bool:
        urls = self._settings.build_camera_urls()
        for url in urls:
            if self._stop.is_set():
                return False
            _LOG.info("카메라 URL 시도: %s", url)
            if self._try_open_url(url):
                return True
        _LOG.warning("모든 카메라 URL 연결 실패. 후보: %s", urls)
        return False

    def _store_frame(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        try:
            if self._settings.FRAME_WIDTH > 0 and self._settings.FRAME_HEIGHT > 0:
                h, w = frame.shape[:2]
                if (w, h) != (self._settings.FRAME_WIDTH, self._settings.FRAME_HEIGHT):
                    frame = cv2.resize(
                        frame,
                        (self._settings.FRAME_WIDTH, self._settings.FRAME_HEIGHT),
                        interpolation=cv2.INTER_AREA,
                    )
        except Exception as e:  # noqa: BLE001
            _LOG.debug("프레임 리사이즈 생략: %s", e)

        with self._lock:
            self._latest_frame = frame.copy()
            self._last_frame_at = _utc_now()
            self._frame_size = (int(frame.shape[1]), int(frame.shape[0]))

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            if self._cap is None or not self._connected:
                self._release_cap_safe()
                ok = self._connect_any()
                if not ok:
                    self._reconnect_count += 1
                    if (
                        self._settings.MAX_RECONNECT_ATTEMPTS > 0
                        and self._reconnect_count >= self._settings.MAX_RECONNECT_ATTEMPTS
                    ):
                        self._last_error = "max reconnect attempts reached (will retry later)"
                        _LOG.error(
                            "카메라 재연결 상한(%s) 도달 — %ss 후 재시도",
                            self._settings.MAX_RECONNECT_ATTEMPTS,
                            self._settings.RECONNECT_DELAY_SECONDS * 2,
                        )
                        time.sleep(self._settings.RECONNECT_DELAY_SECONDS * 2)
                        self._reconnect_count = 0
                    else:
                        time.sleep(self._settings.RECONNECT_DELAY_SECONDS)
                    continue

            assert self._cap is not None
            with self._lock:
                ts = self._last_frame_at
            if ts is not None and (_utc_now() - ts).total_seconds() > self._settings.FRAME_STALE_SECONDS:
                _LOG.warning("마지막 프레임 시각 stale — 재연결")
                self._release_cap_safe()
                time.sleep(self._settings.RECONNECT_DELAY_SECONDS)
                continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._read_fail_count += 1
                self._read_fail_streak += 1
                _LOG.debug("프레임 read 실패 (streak=%s)", self._read_fail_streak)
                if self._read_fail_streak >= self._settings.CONSECUTIVE_READ_FAILS_BEFORE_RECONNECT:
                    _LOG.warning("연속 read 실패 임계 — 재연결")
                    self._release_cap_safe()
                    self._read_fail_streak = 0
                    time.sleep(self._settings.RECONNECT_DELAY_SECONDS)
                    continue
                time.sleep(0.01)
                continue

            self._read_fail_streak = 0

            try:
                self._store_frame(frame)
            except Exception as e:  # noqa: BLE001
                _LOG.exception("프레임 저장 오류: %s", e)

        self._release_cap_safe()

    def get_latest_frame_copy(self) -> np.ndarray | None:
        """추론 루프용 최신 프레임 복사."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            last_at = self._last_frame_at
            fw, fh = self._frame_size if self._frame_size else (None, None)
        stale = False
        if last_at is not None:
            stale = (_utc_now() - last_at).total_seconds() > self._settings.FRAME_STALE_SECONDS
        return {
            "connected": bool(self._connected and self._cap is not None),
            "active_url": self._active_url,
            "candidate_urls": self._settings.build_camera_urls(),
            "reconnect_count": self._reconnect_count,
            "read_fail_count": self._read_fail_count,
            "last_frame_at": last_at.isoformat().replace("+00:00", "Z") if last_at else None,
            "frame_width": fw,
            "frame_height": fh,
            "stale": stale,
            "error_message": self._last_error,
            "camera_ready": self._connected and last_at is not None and not stale,
            "opencv_available": cv2 is not None,
        }
