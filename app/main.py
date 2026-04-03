"""
FastAPI 엔트리포인트.

실시간 카메라 + YOLO 추론은 lifespan 에서 백그라운드로 기동한다.
모델/카메라 실패 시에도 HTTP 서버는 기동한다.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings
from app.services.camera_service import CameraService
from app.services.detection_loop import InferenceLoop
from app.services.detection_service import DetectionService
from app.services.stream_service import StreamService
from app.utils.logger import get_logger, setup_logging

setup_logging(get_settings().LOG_LEVEL)
_LOG = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)

    app.state.started_at = time.time()
    app.state.app_name = settings.APP_NAME
    app.state.app_version = settings.APP_VERSION

    camera = CameraService(settings)
    detection = DetectionService(settings)
    stream = StreamService(settings, camera, detection)
    inference_loop = InferenceLoop(settings, camera, detection)

    app.state.camera = camera
    app.state.detection = detection
    app.state.stream = stream
    app.state.inference_loop = inference_loop

    _LOG.info("서버 시작: %s v%s", settings.APP_NAME, settings.APP_VERSION)
    _LOG.info("MODEL_PATH=%s", settings.MODEL_PATH)

    try:
        detection.try_load()
    except Exception as e:  # noqa: BLE001
        _LOG.exception("모델 로드 단계 예외(무시하고 기동): %s", e)

    # DISPLAY_MODE=2: 화면 캡처(mss)만 사용 — DroidCam 등 카메라 연결 불필요
    if int(settings.DISPLAY_MODE) == 2:
        _LOG.info("DISPLAY_MODE=2 — 카메라 스트림을 시작하지 않음(화면 갈무리 추론)")
    else:
        try:
            camera.start()
        except Exception as e:  # noqa: BLE001
            _LOG.exception("카메라 시작 예외(무시하고 기동): %s", e)

    try:
        inference_loop.start()
    except Exception as e:  # noqa: BLE001
        _LOG.exception("추론 루프 시작 예외(무시하고 기동): %s", e)

    yield

    try:
        inference_loop.stop()
    except Exception as e:  # noqa: BLE001
        _LOG.exception("추론 루프 중지 예외: %s", e)
    if int(settings.DISPLAY_MODE) != 2:
        try:
            camera.stop()
        except Exception as e:  # noqa: BLE001
            _LOG.exception("카메라 중지 예외: %s", e)
    _LOG.info("서버 종료 처리 완료")


settings = get_settings()

app = FastAPI(
    title="AiSilverBridgeSky Detection API",
    version=settings.APP_VERSION,
    description="DroidCam 실시간 스트림 + YOLO 감지 + 상태 API",
    lifespan=lifespan,
)

app.include_router(api_router)
