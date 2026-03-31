# Training — Fire / Smoke / Knife 통합 (YOLO26)

## 목표

- `archive.zip`(DFire) + `DFS-Fire.yolo26.zip`(DFS)를 **fire / smoke 2클래스**로 통합해 `training/datasets/detect/` 에 둔 뒤, **YOLO26** 으로 학습한다.
- (기본) 2클래스 설정에서는 person / knife / fall 은 다루지 않는다.

추가로 `Knifedataset.zip` 를 합쳐 **fire / smoke / knife 3클래스** 데이터셋을 만들고, 학습을 이어갈 수 있는 구조도 준비되어 있다.

## 사전 설치

필수:

```bash
pip install ultralytics pyyaml
```

선택:

```bash
pip install tqdm
```

또는:

```bash
cd /home/apps/SilverBridgeSky/AiSilverBridgeSky
pip install -r training/requirements-train.txt
```

## Zip 위치 (기본값)

| 파일 | 기본 경로 |
|------|-----------|
| DFire | `training/datasets/archive.zip` |
| DFS | `training/datasets/DFS-Fire.yolo26.zip` |

압축 해제 위치:

- `training/datasets/_sources/dfire/`
- `training/datasets/_sources/dfs_fire/`

이미 풀려 있으면 **자동으로 스킵** (`--force-extract` 로 재해제).

## 통합 데이터셋 준비

```bash
cd /home/apps/SilverBridgeSky/AiSilverBridgeSky

python3 training/scripts/prepare_fire_smoke_dataset.py --clean-detect
```

- `--clean-detect`: `training/datasets/detect/` 아래 train/val 이미지·라벨을 비운 뒤 다시 채움
- `--dry-run`: `detect/` 에 쓰지 않고 로그만 (압축 해제는 수행)
- `--zip-dfire`, `--zip-dfs`: zip 경로 오버라이드

### 동작 요약

1. 두 zip 존재 확인 후 `_sources/` 에 해제
2. 각 `data.yaml` 로 클래스 이름 로드 → **fire→0, smoke→1**, **other·불명** 라벨 줄 제거
3. **DFire** 이미지는 `dfire_<원본파일명>` 으로 복사, **DFS** 는 `dfs_<원본파일명>` (prefix 로 덮어쓰기 방지)
4. DFS zip 에 **val 폴더가 없는 경우** train 목록에서 약 **10%** 를 val 로 분할
5. `training/datasets/detect/data.yaml` 새로 생성
6. 검증 후 요약 출력

### 최종 디렉터리

```
training/datasets/detect/
├── data.yaml
├── images/train/
├── images/val/
├── labels/train/
└── labels/val/
```

## Knife 추가 (fire/smoke/knife 3클래스)

### 데이터셋 준비

`Knifedataset.zip` 까지 포함해 `training/datasets/detect_fs_knife/` 로 통합한다.

```bash
cd /home/apps/SilverBridgeSky/AiSilverBridgeSky
python3 training/scripts/prepare_fire_smoke_knife_dataset.py --clean-detect
```

주요 결과:
```
training/datasets/detect_fs_knife/
├── data.yaml   # names: fire=0, smoke=1, knife=2
├── images/train/
├── images/val/
├── labels/train/
└── labels/val/
```

### 3클래스 학습(추가학습/이어학습)

현재 돌아가고 있는 2클래스 학습이 끝난 뒤, 아래처럼 `--data-yaml`과 `--weights`(또는 `--resume-from`)를 지정해서 이어학습한다.

예) `best.pt`에서 시작해 3클래스로 학습:

```bash
cd /home/apps/SilverBridgeSky/AiSilverBridgeSky
python3 training/scripts/train_yolo.py \
  --device 0 \
  --data-yaml training/datasets/detect_fs_knife/data.yaml \
  --weights training/runs/fire_smoke_yolo26/weights/best.pt \
  --name fire_smoke_knife_yolo26 \
  --epochs 20 \
  --exist-ok
```

## 학습

```bash
cd /home/apps/SilverBridgeSky/AiSilverBridgeSky
python3 training/scripts/train_yolo.py --device 0
```

| 인자 | 기본값 |
|------|--------|
| `--epochs` | 100 |
| `--batch` | 16 |
| `--imgsz` | 640 |
| `--device` | 0 |
| `--workers` | 8 |
| `--project` | `training/runs` |
| `--name` | `fire_smoke_yolo26` |

추가학습(이어학습)도 같은 `name` 디렉터리의 `weights/last.pt` 또는 `weights/best.pt`에서 시작할 수 있다.

### 추가학습 예시

1. 현재 학습 상태를 기반으로 “optimizer까지” 이어서 학습

```bash
cd /home/apps/SilverBridgeSky/AiSilverBridgeSky
python3 training/scripts/train_yolo.py --device 0 --resume-optim --resume-from last --epochs 20 --exist-ok
```

2. best 가중치에서 파인튜닝(옵티마이저 상태는 이어가지 않을 수 있음)

```bash
cd /home/apps/SilverBridgeSky/AiSilverBridgeSky
python3 training/scripts/train_yolo.py --device 0 --resume-from best --epochs 20 --exist-ok
```

### 결과 가중치

```
/home/apps/SilverBridgeSky/AiSilverBridgeSky/training/runs/fire_smoke_yolo26/weights/best.pt
```

## 원본 구조 메모

`training/DATASET_STRUCTURE_NOTES.md` 에 다운로드 경로·특이사항을 적어 두면 재현에 도움이 된다.

## 제외

- 서비스 `service/` 추론 API 연동
- fall / person 클래스 (3클래스 단계에서는 knife만 추가)
