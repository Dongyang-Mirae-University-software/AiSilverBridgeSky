"""
Ultralytics YOLO 로딩·추론·결과 캐시.

모델이 없거나 추론 실패해도 프로세스는 유지한다.
"""

from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.core.config import Settings
from app.utils.class_mapper import build_model_class_summary, normalize_class_name
from app.utils.image_utils import overlay_model_unavailable
from app.utils.logger import get_logger

_LOG = get_logger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[misc, assignment]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


class DetectionService:
    """단일 YOLO 모델, 스레드 안전한 최신 추론 결과."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._model: Any = None
        self._model_loaded = False
        self._load_error: str | None = None
        self._raw_class_names: list[str] = []
        self._summary: dict[str, Any] = build_model_class_summary([])

        self._latest_detections: list[dict[str, Any]] = []
        self._latest_annotated: np.ndarray | None = None
        self._last_inference_at: datetime | None = None
        self._last_inference_ms: float | None = None
        self._last_error: str | None = None
        self._infer_count = 0
        self._infer_ms_ema: float | None = None

    def try_load(self) -> None:
        """시작 시 1회 호출. 실패해도 예외를 밖으로 던지지 않는다."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self._load_error = "ultralytics 미설치"
            _LOG.error(self._load_error)
            return

        path = self._settings.MODEL_PATH
        if not path.is_file():
            self._load_error = f"모델 파일 없음: {path}"
            _LOG.warning(self._load_error)
            return

        try:
            model = YOLO(str(path))
            self._model = model
            self._model_loaded = True
            self._load_error = None
            names = getattr(model, "names", None)
            raw: list[str] = []
            if isinstance(names, dict):
                try:
                    keys = sorted(int(k) for k in names.keys())
                    raw = [str(names[i]) for i in keys]
                except Exception:
                    raw = [str(v) for v in names.values()]
            elif isinstance(names, (list, tuple)):
                raw = [str(x) for x in names]
            elif names is not None:
                raw = [str(names)]

            self._raw_class_names = raw
            self._summary = build_model_class_summary(raw)
            _LOG.info("YOLO 모델 로드 성공: %s classes=%s", path, raw)
        except Exception as e:  # noqa: BLE001
            self._model = None
            self._model_loaded = False
            self._load_error = str(e)
            _LOG.exception("YOLO 모델 로드 실패: %s", e)

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def model_path(self) -> str:
        return str(self._settings.MODEL_PATH)

    def get_class_summary(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._summary)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "model_loaded": self._model_loaded,
                "model_path": self.model_path,
                "load_error": self._load_error,
                "summary": dict(self._summary),
                "last_inference_at": self._last_inference_at.isoformat().replace("+00:00", "Z")
                if self._last_inference_at
                else None,
                "last_inference_ms": self._last_inference_ms,
                "last_error": self._last_error,
                "avg_inference_ms_ema": self._infer_ms_ema,
                "inference_count": self._infer_count,
            }

    def run_inference(self, frame_bgr: np.ndarray | None) -> None:
        """최신 프레임 1장 처리. None이면 스킵."""
        if frame_bgr is None or frame_bgr.size == 0:
            return
        if not self._model_loaded or self._model is None:
            with self._lock:
                self._latest_detections = []
                self._latest_annotated = overlay_model_unavailable(frame_bgr) if cv2 else frame_bgr.copy()
                self._last_inference_at = _utc_now()
                self._last_inference_ms = 0.0
                self._last_error = self._load_error or "model not loaded"
            return

        t0 = time.perf_counter()
        try:
            results = self._model.predict(
                source=frame_bgr,
                conf=self._settings.CONF_THRESHOLD,
                iou=self._settings.IOU_THRESHOLD,
                verbose=False,
            )
        except Exception as e:  # noqa: BLE001
            _LOG.exception("YOLO predict 오류: %s", e)
            with self._lock:
                self._last_error = str(e)
            return

        dt_ms = (time.perf_counter() - t0) * 1000.0
        detections: list[dict[str, Any]] = []
        annotated = frame_bgr.copy()

        try:
            if not results:
                raise ValueError("empty results")
            r0 = results[0]
            if r0 is None:
                raise ValueError("null result")
            names = getattr(self._model, "names", {}) or {}
            plot_fn = getattr(r0, "plot", None)
            if plot_fn is not None and cv2 is not None:
                try:
                    plotted = plot_fn()
                    if isinstance(plotted, np.ndarray) and plotted.size > 0:
                        annotated = plotted
                except Exception:
                    pass

            boxes = getattr(r0, "boxes", None)
            if boxes is None:
                xyxy, conf, cls = None, None, None
            else:
                xyxy = getattr(boxes, "xyxy", None)
                conf = getattr(boxes, "conf", None)
                cls = getattr(boxes, "cls", None)
            if xyxy is None:
                n = 0
            else:
                n = int(len(xyxy))
            for i in range(n):
                c = int(cls[i].item()) if cls is not None and i < len(cls) else -1
                raw_name = ""
                if isinstance(names, dict) and c in names:
                    raw_name = str(names[c])
                elif isinstance(names, dict):
                    raw_name = str(names.get(c, ""))
                elif isinstance(names, (list, tuple)) and 0 <= c < len(names):
                    raw_name = str(names[c])
                else:
                    raw_name = str(c)

                display = normalize_class_name(raw_name)
                score = _safe_float(conf[i].item(), 0.0) if conf is not None and i < len(conf) else 0.0
                x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[i].tolist()]
                now = _utc_now()
                detections.append(
                    {
                        "class_id": c,
                        "original_class_name": raw_name,
                        "display_class_name": display,
                        "confidence": score,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "detected_at": now.isoformat().replace("+00:00", "Z"),
                    }
                )

            # 간단 오버레이: FPS / latency
            if cv2 is not None:
                label = f"{dt_ms:.1f} ms | dets={len(detections)}"
                cv2.putText(
                    annotated,
                    label,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (80, 220, 80),
                    2,
                    cv2.LINE_AA,
                )
        except Exception as e:  # noqa: BLE001
            _LOG.exception("결과 파싱 오류: %s", e)
            with self._lock:
                self._last_error = str(e)
            return

        with self._lock:
            self._latest_detections = detections
            self._latest_annotated = annotated
            self._last_inference_at = _utc_now()
            self._last_inference_ms = dt_ms
            self._last_error = None
            self._infer_count += 1
            if self._infer_ms_ema is None:
                self._infer_ms_ema = dt_ms
            else:
                self._infer_ms_ema = 0.9 * self._infer_ms_ema + 0.1 * dt_ms

    def get_latest_for_api(self) -> tuple[list[dict[str, Any]], np.ndarray | None, dict[str, Any]]:
        """API 응답용 스냅샷."""
        with self._lock:
            dets = list(self._latest_detections)
            ann = self._latest_annotated.copy() if self._latest_annotated is not None else None
            meta = {
                "last_inference_at": self._last_inference_at.isoformat().replace("+00:00", "Z")
                if self._last_inference_at
                else None,
                "last_inference_ms": self._last_inference_ms,
                "last_error": self._last_error,
                "model_loaded": self._model_loaded,
            }
        return dets, ann, meta

    def get_annotated_or_raw(self, raw_frame: np.ndarray | None) -> np.ndarray | None:
        """스트림용: annotated 우선, 없으면 raw, 없으면 None."""
        with self._lock:
            ann = self._latest_annotated
            if ann is not None:
                return ann.copy()
        return raw_frame.copy() if raw_frame is not None else None
