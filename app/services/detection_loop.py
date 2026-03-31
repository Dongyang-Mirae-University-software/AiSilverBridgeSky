"""카메라 최신 프레임을 일정 주기로만 추론 (큐 적재 없음)."""

from __future__ import annotations

import os
import threading
import time

import numpy as np

from app.core.config import Settings
from app.services.camera_service import CameraService
from app.services.detection_service import DetectionService
from app.utils.logger import get_logger

_LOG = get_logger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[misc, assignment]


class InferenceLoop:
    """백그라운드에서 inference interval 마다 최신 프레임만 처리."""

    def __init__(
        self,
        settings: Settings,
        camera: CameraService,
        detection: DetectionService,
    ) -> None:
        self._settings = settings
        self._camera = camera
        self._detection = detection
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_loop_error: str | None = None
        self._display_mode = int(settings.DISPLAY_MODE)
        self._display_enabled = self._display_mode == 1
        self._screen_capture_enabled = self._display_mode == 2
        self._display_failed_once = False
        self._display_prechecked = False
        self._mss = None
        self._capture_failed_once = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="inference-loop", daemon=True)
        self._thread.start()
        _LOG.info("추론 루프 시작 (interval=%sms, display_mode=%s)", self._settings.INFERENCE_INTERVAL_MS, self._display_mode)
        if self._display_enabled:
            _LOG.info(
                "로컬 디스플레이 활성화: window=%s source=%s",
                self._settings.DISPLAY_WINDOW_NAME,
                self._settings.DISPLAY_SOURCE,
            )
        if self._screen_capture_enabled:
            _LOG.info("화면 갈무리 추론 모드 활성화: region=%s", self._settings.SCREEN_CAPTURE_REGION)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._close_window()
        _LOG.info("추론 루프 중지")

    @property
    def last_loop_error(self) -> str | None:
        return self._last_loop_error

    def _close_window(self) -> None:
        if not self._display_enabled or cv2 is None:
            return
        try:
            cv2.destroyWindow(self._settings.DISPLAY_WINDOW_NAME)
        except Exception:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _precheck_display(self) -> None:
        """
        imshow 호출 전 최소 안전검사.
        Qt/xcb 환경에서 display 미접속 상태로 imshow를 호출하면
        Python 예외가 아니라 프로세스 abort가 발생할 수 있어 선제 차단한다.
        """
        if self._display_prechecked:
            return
        self._display_prechecked = True

        if not self._display_enabled:
            return
        if cv2 is None:
            self._display_enabled = False
            return

        display = os.environ.get("DISPLAY", "").strip()
        if not display:
            _LOG.warning("DISPLAY_ENABLED=true 이지만 DISPLAY 환경변수가 없어 비활성화")
            self._display_enabled = False
            return

        # :0, :1.0 형태를 /tmp/.X11-unix/XN 소켓으로 매핑해 기본 접근성 확인
        # 소켓이 없으면 imshow 시 프로세스 abort 가능성이 높다.
        disp_head = display.split(".")[0]
        if disp_head.startswith(":"):
            xnum = disp_head[1:]
            if xnum.isdigit():
                sock = f"/tmp/.X11-unix/X{xnum}"
                if not os.path.exists(sock):
                    _LOG.warning(
                        "DISPLAY=%s 이지만 X 소켓(%s) 없음 — 디스플레이 출력 비활성화",
                        display,
                        sock,
                    )
                    self._display_enabled = False
                    return

    def _render_debug_window(self, raw_frame) -> None:
        if not self._display_enabled:
            return
        if cv2 is None:
            if not self._display_failed_once:
                _LOG.warning("DISPLAY_ENABLED=true 이지만 opencv HighGUI 사용 불가")
                self._display_failed_once = True
            self._display_enabled = False
            return

        show_frame = None
        try:
            _dets, ann, _meta = self._detection.get_latest_for_api()
            src = self._settings.DISPLAY_SOURCE
            if src == "raw":
                show_frame = raw_frame
            elif src == "auto":
                show_frame = ann if ann is not None else raw_frame
            else:
                show_frame = ann if ann is not None else raw_frame

            if show_frame is None:
                return

            cv2.imshow(self._settings.DISPLAY_WINDOW_NAME, show_frame)
            # waitKey 호출이 있어야 HighGUI 이벤트가 처리된다.
            cv2.waitKey(1)
        except Exception as e:  # noqa: BLE001
            if not self._display_failed_once:
                _LOG.warning("디스플레이 출력 비활성화: %s", e)
                self._display_failed_once = True
            self._display_enabled = False

    def _capture_screen_frame(self):
        if not self._screen_capture_enabled:
            return None
        if cv2 is None:
            if not self._capture_failed_once:
                _LOG.warning("screen-capture 모드이지만 cv2 미설치")
                self._capture_failed_once = True
            return None
        try:
            if self._mss is None:
                import mss  # type: ignore

                self._mss = mss.mss()
            monitor = self._mss.monitors[1]
            region = self._settings.SCREEN_CAPTURE_REGION
            if region is not None:
                x, y, w, h = region
                monitor = {"left": x, "top": y, "width": w, "height": h}
            shot = self._mss.grab(monitor)
            frame = np.array(shot)
            # BGRA -> BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:  # noqa: BLE001
            if not self._capture_failed_once:
                _LOG.warning("화면 갈무리 실패: %s", e)
                self._capture_failed_once = True
            return None

    def _run(self) -> None:
        interval = max(0.01, self._settings.INFERENCE_INTERVAL_MS / 1000.0)
        self._precheck_display()
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                if self._screen_capture_enabled:
                    frame = self._capture_screen_frame()
                else:
                    frame = self._camera.get_latest_frame_copy()
                self._detection.run_inference(frame)
                self._render_debug_window(frame)
                self._last_loop_error = None
            except Exception as e:  # noqa: BLE001
                self._last_loop_error = str(e)
                _LOG.exception("추론 루프 예외: %s", e)
            elapsed = time.perf_counter() - t0
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
