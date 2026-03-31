"""클래스 요약·최근 감지 결과."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.schemas.detection import (
    ClassesResponse,
    DetectionCurrentResponse,
    DetectionItemOut,
    BBox,
)
from app.utils.class_mapper import build_model_class_summary

router = APIRouter(tags=["detections"])


@router.get("/classes", response_model=ClassesResponse)
def get_classes(request: Request) -> ClassesResponse:
    det = getattr(request.app.state, "detection", None)
    if det is None:
        summ = build_model_class_summary([])
        return ClassesResponse(**summ)
    summ = det.get_class_summary()
    return ClassesResponse(**summ)


@router.get("/detections/current", response_model=DetectionCurrentResponse)
def get_current_detections(request: Request) -> DetectionCurrentResponse:
    det = getattr(request.app.state, "detection", None)
    cam = getattr(request.app.state, "camera", None)
    frame_ok = False
    if cam is not None:
        f = cam.get_latest_frame_copy()
        frame_ok = f is not None

    if det is None:
        return DetectionCurrentResponse(model_loaded=False, frame_available=frame_ok)

    dets, _ann, meta = det.get_latest_for_api()
    out_items: list[DetectionItemOut] = []
    for d in dets:
        bb = d.get("bbox") or {}
        try:
            out_items.append(
                DetectionItemOut(
                    class_id=int(d.get("class_id", -1)),
                    original_class_name=str(d.get("original_class_name", "")),
                    display_class_name=str(d.get("display_class_name", "")),
                    confidence=float(d.get("confidence", 0.0)),
                    bbox=BBox(
                        x1=int(bb.get("x1", 0)),
                        y1=int(bb.get("y1", 0)),
                        x2=int(bb.get("x2", 0)),
                        y2=int(bb.get("y2", 0)),
                    ),
                    detected_at=d.get("detected_at"),
                )
            )
        except Exception:
            continue

    return DetectionCurrentResponse(
        detected_at=meta.get("last_inference_at"),
        inference_ms=meta.get("last_inference_ms"),
        total_detections=len(out_items),
        detections=out_items,
        model_loaded=bool(meta.get("model_loaded")),
        frame_available=frame_ok,
        last_error=meta.get("last_error"),
    )
