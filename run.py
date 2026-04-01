#!/usr/bin/env python3
"""
간단 실행 런처.

예시:
  python3 run.py -d 1 -h 192.168.0.18 -p 4747 -P 8000
"""

from __future__ import annotations

import argparse
import os

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    # -h 를 host로 쓰기 위해 add_help=False 사용
    p = argparse.ArgumentParser(
        prog="run.py",
        description="SilverBridge Detection Service launcher",
        add_help=False,
    )
    p.add_argument("--help", action="help", help="show this help message and exit")

    p.add_argument(
        "-d",
        "--display-mode",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="display mode: 0=off, 1=window, 2=screen-capture",
    )
    p.add_argument(
        "-h",
        "--camera-host",
        type=str,
        default="192.168.0.18",
        help="camera host (DroidCam IP)",
    )
    p.add_argument(
        "-p",
        "--camera-port",
        type=int,
        default=4747,
        help="camera port",
    )
    p.add_argument(
        "-P",
        "--port",
        type=int,
        default=8000,
        help="api server port",
    )
    p.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="MODEL_PATH override (기본: .env/.env.example 값 사용)",
    )
    p.add_argument(
        "--camera-paths",
        type=str,
        default="/video,/mjpegfeed,/",
        help="CAMERA_PATH_CANDIDATES csv",
    )
    p.add_argument(
        "--display-source",
        type=str,
        default="annotated",
        choices=["annotated", "raw", "auto"],
        help="DISPLAY_SOURCE when display mode is 1",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.model is not None and str(args.model).strip():
        os.environ["MODEL_PATH"] = str(args.model).strip()
    os.environ["CAMERA_HOST"] = str(args.camera_host)
    os.environ["CAMERA_PORT"] = str(args.camera_port)
    os.environ["CAMERA_PATH_CANDIDATES"] = str(args.camera_paths)
    os.environ["DISPLAY_MODE"] = str(args.display_mode)
    os.environ["DISPLAY_SOURCE"] = str(args.display_source)

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(args.port),
        reload=False,
    )


if __name__ == "__main__":
    main()

