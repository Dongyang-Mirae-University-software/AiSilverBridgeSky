# `training/datasets/` 구조

이 디렉터리 아래 **이미지·라벨·압축 해제 원본 등 대용량 데이터는 Git에 포함하지 않습니다.**  
저장소에는 **폴더 뼈대(`.gitkeep`)**, **`data.yaml`**, 이 **README** 정도만 올라갑니다.

로컬에서 데이터를 채우려면 `training/README.md` 및 준비 스크립트를 참고하세요.

예상 레이아웃(개략):

- `detect/` — fire/smoke 통합 검출용
- `detect_fs_knife/` — fire/smoke/knife 통합 검출용
- `fall/` — 낙상 관련(원시·가공·어노테이션)
- `_sources/` — zip 해제 등 중간 산출(로컬 전용)
