#!/usr/bin/env python3
"""
Ultralytics YOLO26 기반 fire/smoke 2-class 학습.

사전 조건:
  - prepare_fire_smoke_dataset.py 로 datasets/detect 정리 완료
  - data.yaml, yolo26n.pt 존재
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path("/home/apps/SilverBridgeSky/AiSilverBridgeSky")
MODEL_PATH = ROOT / "yolo26n.pt"
DEFAULT_DATA_YAML = ROOT / "training" / "datasets" / "detect" / "data.yaml"
PROJECT_DIR = ROOT / "training" / "runs"

RUN_WEIGHTS_DIR = lambda project, name: Path(project) / name / "weights"


def validate_before_train(data_yaml: Path) -> None:
    """학습 직전 필수 경로 검증."""
    err: list[str] = []
    if not MODEL_PATH.is_file():
        err.append(f"모델 가중치 없음: {MODEL_PATH}")
    if not data_yaml.is_file():
        err.append(f"data.yaml 없음: {data_yaml}")
    base = data_yaml.parent
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        p = base / sub
        if not p.is_dir():
            err.append(f"디렉터리 없음: {p}")

    img_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    tr_dir, va_dir = base / "images/train", base / "images/val"
    n_train = sum(1 for f in tr_dir.glob("*") if f.is_file() and f.suffix.lower() in img_ext) if tr_dir.is_dir() else 0
    n_val = sum(1 for f in va_dir.glob("*") if f.is_file() and f.suffix.lower() in img_ext) if va_dir.is_dir() else 0
    if n_train == 0:
        err.append("images/train 에 학습 이미지가 없습니다.")
    if n_val == 0:
        err.append("images/val 에 검증 이미지가 없습니다.")

    if err:
        raise RuntimeError("사전 검증 실패:\n  - " + "\n  - ".join(err))


def resolve_weights(args: argparse.Namespace) -> Path:
    """
    학습 시작 가중치 경로를 결정한다.

    - --resume-from base: yolo26n.pt
    - --resume-from last/best: 기존 run의 weights에서 선택
    - --resume-from auto: last 우선, 없으면 best
    - --weights: 최우선 (직접 경로 지정)
    """
    if args.weights is not None:
        p = Path(args.weights).expanduser().resolve()
        if not p.is_file():
            raise RuntimeError(f"--weights 경로가 파일이 아닙니다: {p}")
        return p

    resume_from = args.resume_from
    if resume_from == "base":
        return MODEL_PATH

    weights_dir = RUN_WEIGHTS_DIR(args.project, args.name)
    if resume_from == "last":
        p = weights_dir / "last.pt"
        if not p.is_file():
            raise RuntimeError(f"last.pt 없음: {p}")
        return p
    if resume_from == "best":
        p = weights_dir / "best.pt"
        if not p.is_file():
            raise RuntimeError(f"best.pt 없음: {p}")
        return p
    if resume_from == "auto":
        p_last = weights_dir / "last.pt"
        p_best = weights_dir / "best.pt"
        if p_last.is_file():
            return p_last
        if p_best.is_file():
            return p_best
        raise RuntimeError(f"auto 선택 실패: {weights_dir} 에 last.pt/best.pt 없음")

    raise RuntimeError(f"알 수 없는 --resume-from 값: {resume_from}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0", help='예: "0", "cpu", "0,1"')
    p.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    p.add_argument("--project", type=Path, default=PROJECT_DIR)
    p.add_argument("--name", type=str, default="fire_smoke_yolo26")
    p.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="학습에 사용할 data.yaml 경로 (fire/smoke/knife 등 클래스 수에 영향)",
    )

    p.add_argument(
        "--resume-from",
        type=str,
        default="base",
        choices=["base", "auto", "best", "last"],
        help="이어학습 시작 가중치 선택",
    )
    p.add_argument(
        "--resume-optim",
        action="store_true",
        help="Ultralytics resume=True 로 옵티마이저 상태까지 이어학습 시도",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="학습 시작 가중치 경로(직접 지정) - --resume-from 보다 우선",
    )
    p.add_argument(
        "--exist-ok",
        action="store_true",
        help="기존 run 디렉터리가 있으면 재사용(기본은 false)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml: Path = args.data_yaml
    try:
        validate_before_train(data_yaml)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics 가 설치되어 있지 않습니다.\n"
            "  pip install -r training/requirements-train.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    weights_path = resolve_weights(args)
    model = YOLO(str(weights_path))

    # resume=True는 last run 상태를 기준으로 이어 학습하는 방식이라서,
    # resume-from이 best인 경우에도 last가 있어야 안정적이다.
    # (여기서는 resume-from=best라도 resume-optim=True면 last를 우선 사용하도록 유도하지 않고,
    #  사용자가 의도한 동작을 그대로 따르되, 에러가 나면 메시지로 안내됨)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=bool(args.exist_ok),
        resume=bool(args.resume_optim),
    )

    weights_dir = Path(args.project) / args.name / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    print("학습 종료.")
    print(f"  best.pt: {best} (존재 여부는 실행 결과에 따름)")
    print(f"  last.pt: {last} (존재 여부는 실행 결과에 따름)")


if __name__ == "__main__":
    main()
