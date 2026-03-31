# SilverBridge — 실시간 감지 API (`service/`)

## 개요

**DroidCam HTTP 스트림**을 받아 **Ultralytics YOLO**로 객체를 감지하는 **FastAPI** 서비스입니다.  
내부망·Docker에서 Spring 등 백엔드가 호출하는 **내부 전용** 용도를 전제로 하며, **모델 파일이 없거나 카메라가 꺼져 있어도 HTTP 서버는 기동**합니다.

- **스택**: FastAPI, OpenCV, Ultralytics YOLO, Uvicorn  
- **실시간성**: 카메라 스레드는 최신 프레임만 유지하고, 추론 루프는 `INFERENCE_INTERVAL_MS` 주기로만 처리합니다(큐 적재 없음).

## 목표 클래스 (4종, 표시·기획 기준)

| 표시명 | 비고 |
|--------|------|
| DFS | 데이터·모델 명명에 따라 다를 수 있음 |
| D-fire | `fire`, `smoke`, `d-fire` 등은 정규화 시 D-fire 로 묶을 수 있음 |
| Knife | 긴 원본 라벨도 `Knife`로 표시 |
| Fall | `fall`, `넘어짐` 등 |

실제 `best.pt`는 **2·3·4클래스 등 학습 진행 단계에 따라 클래스 수가 다를 수 있습니다.**  
서비스는 **모델이 제공하는 `names`만** 사용하며, **클래스 인덱스 고정(0=DFS 등)은 하지 않습니다.**  
미학습 클래스는 `/classes` 응답의 `missing_classes`로만 표시됩니다.

## 주요 동작

- **카메라**: `CAMERA_PATH_CANDIDATES` 순으로 `http://{host}:{port}{path}` 연결, 첫 프레임 성공 시 채택, 끊김·stale 시 재연결  
- **모델**: `MODEL_PATH` 로드 실패 시에도 기동, 추론 생략 + 상태 API로 보고  
- **스트림**: 브라우저에서 `GET /stream` (MJPEG), `GET /frame` (단일 JPEG)

## 디렉터리

| 경로 | 역할 |
|------|------|
| `app/main.py` | FastAPI, lifespan(카메라·추론 스레드) |
| `app/core/config.py` | 환경 변수·기본값 |
| `app/utils/logger.py` | 로깅 |
| `app/utils/class_mapper.py` | 클래스명 정규화 |
| `app/services/camera_service.py` | 스트림·재연결·프레임 캐시 |
| `app/services/detection_service.py` | YOLO·결과 캐시 |
| `app/services/detection_loop.py` | 추론 주기 루프 |
| `app/services/stream_service.py` | MJPEG |
| `app/api/routes/` | REST 엔드포인트 |
| `app/schemas/` | Pydantic 응답 모델 |

## 실행 방법

```bash
cd /path/to/AiSilverBridgeSky/service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 카메라 IP·모델 경로 수정
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 축약 실행 옵션 (`-d -h -p -P`)

```bash
cd /home/apps/SilverBridgeSky/AiSilverBridgeSky/service
python3 run.py -d 1 -h 192.168.0.18 -p 4747 -P 8000 -m /home/apps/SilverBridgeSky/AiSilverBridgeSky/training/runs/fire_smoke_yolo26/weights/best.pt
```

- `-d`: display mode (`0` 끔, `1` 창 표시, `2` 화면 갈무리)
- `-h`: camera host
- `-p`: camera port
- `-P`: API 포트
- `-m`: 모델 경로 (`MODEL_PATH`)
- 도움말: `python3 run.py --help`

개발 시:

```bash
PYTHONPATH=. uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger: `http://127.0.0.1:8000/docs`

## 환경 변수

`.env.example` 참고. 특히 다음을 확인합니다.

- **`MODEL_PATH`**: 학습 산출 `best.pt` (없으면 degraded, 서버는 유지)  
- **`CAMERA_HOST` / `CAMERA_PORT` / `CAMERA_PATH_CANDIDATES`**: DroidCam 주소·경로 후보 (`/video` → `/mjpegfeed` → `/`)  
- **`INFERENCE_INTERVAL_MS`**, **`RECONNECT_DELAY_SECONDS`**, **`FRAME_STALE_SECONDS`**
- **`DISPLAY_MODE`**: `0`(끄기), `1`(창 표시), `2`(화면 갈무리 입력 추론)
- **`DISPLAY_ENABLED`**: 구버전 호환(켜면 mode=1)
- **`DISPLAY_WINDOW_NAME`**, **`DISPLAY_SOURCE`**: 창 제목, 표시 소스(`annotated`/`raw`/`auto`)
- **`SCREEN_CAPTURE_REGION`**: mode=2에서 캡처 영역(`x,y,w,h`)

## API 목록

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 서버·모델·카메라·추론 요약 (일부 실패 시에도 200, `status`: ok/degraded/unavailable) |
| GET | `/classes` | 설정 4종 vs 모델 클래스·정규화·missing |
| GET | `/camera/status` | 연결 URL, 재연결 횟수, stale 등 |
| GET | `/detections/current` | 최근 추론 결과 JSON |
| GET | `/frame` | 최근 1프레임 JPEG (annotated 우선) |
| GET | `/stream` | MJPEG 실시간 스트림 |
| POST | `/detect` | 기존 base64 파이프라인(stub 유지) |

## 문제 해결 체크리스트

1. **DroidCam**이 폰에서 실행 중인지, PC/서버와 **동일 Wi‑Fi**인지 확인합니다.  
2. 브라우저에서 `http://<CAMERA_HOST>:<CAMERA_PORT>/video` 접속이 되는지 확인합니다.  
3. **`MODEL_PATH`** 파일 존재 여부와 GPU/CPU 메모리를 확인합니다.  
4. OpenCV가 스트림을 열 수 있는지(방화벽, 포트 4747) 확인합니다.  
5. **클래스 수가 2·3개여도 정상**입니다. `/classes`의 `missing_classes`만 참고하면 됩니다.
6. GUI 없는 서버(예: headless)에서 `DISPLAY_ENABLED=true`면 자동으로 비활성화될 수 있습니다.

## 로컬 디스플레이 옵션 (3가지)

### 0) 디스플레이 안함

```env
DISPLAY_MODE=0
```

### 1) 디스플레이 연결함 (창 표시)

```env
DISPLAY_MODE=1
DISPLAY_WINDOW_NAME=SilverBridge Live
DISPLAY_SOURCE=annotated
```

- `DISPLAY_SOURCE=annotated`: 박스가 그려진 화면 우선
- `DISPLAY_SOURCE=raw`: 원본 카메라 화면
- `DISPLAY_SOURCE=auto`: annotated 있으면 annotated, 없으면 raw

### 2) 화면 갈무리로 인식

```env
DISPLAY_MODE=2
# 예시: 좌상단(100,100)부터 1280x720 영역
SCREEN_CAPTURE_REGION=100,100,1280,720
```

- mode=2에서는 카메라 대신 **스크린 캡처 프레임**을 추론 입력으로 사용합니다.
- `SCREEN_CAPTURE_REGION`이 비어 있으면 메인 화면 전체를 캡처합니다.

## 라이선스·보안

내부 서비스 전제이며, 외부 공개용이 아닙니다.
