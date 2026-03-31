"""루트 라우터 조합."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import camera, detect, detections, health, stream

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(camera.router)
api_router.include_router(detections.router)
api_router.include_router(stream.router)
api_router.include_router(detect.router)
