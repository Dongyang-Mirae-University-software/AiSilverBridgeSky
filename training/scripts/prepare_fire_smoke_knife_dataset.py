#!/usr/bin/env python3
"""
DFire + DFS + Knifedataset を fire/smoke/knife 3-class YOLO 포맷으로 통합한다.

이번 단계에서는 오직 데이터셋 정리(복사/라벨 재작성)와 data.yaml 생성을 목표로 한다.
학습/추론 로직은 포함하지 않는다.

클래스 고정:
  fire  -> 0
  smoke -> 1
  knife -> 2

라벨 라인 처리:
  - YOLO txt 포맷: class_id x_center y_center width height
  - fire/smoke/knife 외 class_id는 제거
  - bbox 값이 [0,1] 범위를 벗어나면 해당 라인은 스킵
  - 라벨 파일은 비어있을 수 있음 (negative sample 허용)
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import yaml

# --- 고정 경로 (지시서의 2-class와 동일한 패턴) ---
ROOT = Path("/home/apps/SilverBridgeSky/AiSilverBridgeSky")
DATASETS = ROOT / "training" / "datasets"

DEFAULT_ZIP_DFIRE = DATASETS / "archive.zip"
DEFAULT_ZIP_DFS = DATASETS / "DFS-Fire.yolo26.zip"
DEFAULT_ZIP_KNIFE = DATASETS / "Knifedataset.zip"

SOURCES_DIR = DATASETS / "_sources"
EXTRACT_DFIRE = SOURCES_DIR / "dfire"
EXTRACT_DFS = SOURCES_DIR / "dfs_fire"
EXTRACT_KNIFE = SOURCES_DIR / "knife"

DEST_3CLASS = DATASETS / "detect_fs_knife"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FINAL_CLASSES: Dict[str, int] = {"fire": 0, "smoke": 1, "knife": 2}


@dataclass
class SplitDirs:
    train_images: Path
    train_labels: Path
    val_images: Path
    val_labels: Path
    need_internal_split: bool = False


@dataclass
class SourceInfo:
    name: str
    extracted_root: Path
    yaml_path: Path
    original_class_names: Dict[int, str]
    remap: Dict[int, Optional[int]]


@dataclass
class Stats:
    train_images_copied: int = 0
    val_images_copied: int = 0
    labels_rewritten: int = 0
    label_lines_kept: int = 0
    label_lines_dropped: int = 0
    label_lines_invalid_bbox: int = 0
    unknown_class_ids: Set[int] = field(default_factory=set)
    empty_label_files: int = 0
    skipped_missing_label: int = 0
    warnings: List[str] = field(default_factory=list)


def extract_zip(zip_path: Path, dest_dir: Path, *, force_extract: bool) -> None:
    """이미 풀린 경우 스킵(재해제는 --force-extract로만)."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"zip 없음: {zip_path}")

    marker = dest_dir / ".extracted_ok"
    if dest_dir.is_dir() and marker.is_file() and not force_extract:
        logging.info("[SKIP] 이미 해제됨: %s", dest_dir)
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    # force_extract 시에만 기존 디렉토리 정리(데이터 손상 방지 위해 marker 기반)
    if force_extract and dest_dir.exists():
        for p in dest_dir.iterdir():
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)

    logging.info("[EXTRACT] %s -> %s", zip_path, dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    marker.write_text("ok\n", encoding="utf-8")


def find_yaml_file(dataset_root: Path) -> Optional[Path]:
    for name in ("data.yaml", "dataset.yaml", "data.yml"):
        p = dataset_root / name
        if p.is_file():
            return p
    return None


def find_dataset_root(start_dir: Path) -> Path:
    """압축 내부에서 yaml이 있는 디렉토리를 최대한 찾아준다."""
    start_dir = start_dir.resolve()
    yml = find_yaml_file(start_dir)
    if yml:
        return start_dir

    subdirs = [p for p in start_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return find_dataset_root(subdirs[0])

    for y in start_dir.rglob("data.yaml"):
        return y.parent
    for y in start_dir.rglob("dataset.yaml"):
        return y.parent
    return start_dir


def load_class_names(yaml_path: Path) -> Dict[int, str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    names = raw.get("names")
    if names is None:
        return {}
    if isinstance(names, dict):
        return {int(k): str(v).strip() for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n).strip() for i, n in enumerate(names)}
    raise ValueError(f"names 형식 오류: {type(names)} @ {yaml_path}")


def normalize_class_name(name: str) -> Optional[str]:
    """
    원본 class 이름을 최종 카테고리로 매핑하기 위한 정규화.

    반환:
      - 'fire', 'smoke', 'knife' : 보존
      - 'other' 또는 None : 제거
    """
    if name is None:
        return None
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 _-]", "", s)
    s = s.replace("-", " ").strip()

    if s in ("smokes", "smoking"):
        s = "smoke"
    if s in ("fires", "flames", "flame"):
        s = "fire"

    if "knife" in s:
        return "knife"
    if "smoke" in s:
        return "smoke"
    if "fire" in s:
        return "fire"
    if s.startswith("other") or s in ("person", "people", "background", "bg"):
        return None
    # 알 수 없는 클래스도 제거 (엄격하게)
    return None


def build_remap(original_class_names: Dict[int, str]) -> Dict[int, Optional[int]]:
    remap: Dict[int, Optional[int]] = {}
    for oid, raw_name in original_class_names.items():
        cat = normalize_class_name(raw_name)
        if cat is None:
            remap[oid] = None
        else:
            remap[oid] = FINAL_CLASSES[cat]
    return remap


def _resolve_yaml_base(yaml_path: Path, raw: Dict[str, Any]) -> Path:
    base = yaml_path.parent
    path_field = raw.get("path")
    if path_field is None:
        return base.resolve()
    p = Path(str(path_field))
    if p.is_absolute():
        # Kaggle/외부의 잘못된 절대경로일 수 있으므로 실제 디렉토리인지 확인
        if p.is_dir():
            return p.resolve()
        return base.resolve()
    return (yaml_path.parent / p).resolve()


def infer_label_dir_from_image_dir(image_dir: Path) -> Path:
    """
    .../images/train -> .../labels/train 형태를 우선으로 추론한다.
    """
    parts = list(image_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).resolve()
    return (image_dir.parent / "labels").resolve()


def iter_images(folder: Path) -> Iterator[Path]:
    if not folder.is_dir():
        return
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def partition_train_val(image_paths: List[Path], val_ratio: float = 0.1) -> Tuple[List[Path], List[Path]]:
    """val 폴더가 없는 경우 train만으로 val을 만든다."""
    paths = sorted(image_paths, key=lambda x: x.name)
    n = len(paths)
    if n <= 1:
        return paths, paths
    val_n = max(1, int(n * val_ratio))
    val_n = min(val_n, n - 1)
    return paths[:-val_n], paths[-val_n:]


def discover_split_dirs(dataset_root: Path, yaml_path: Path) -> SplitDirs:
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base = _resolve_yaml_base(yaml_path, raw)
    train_rel = raw.get("train") or raw.get("train_path")
    val_rel = raw.get("val") or raw.get("valid") or raw.get("validation")

    train_img: Optional[Path] = None
    val_img: Optional[Path] = None

    # 1) YAML로부터 먼저 시도
    if train_rel:
        tr = str(train_rel).replace("\\", "/").strip()
        candidates = [
            (base / tr).resolve(),
            (yaml_path.parent / tr).resolve(),
            (dataset_root / tr).resolve(),
        ]
        train_img = next((c for c in candidates if c.is_dir()), None)

    if val_rel:
        vr = str(val_rel).replace("\\", "/").strip()
        candidates = [
            (base / vr).resolve(),
            (yaml_path.parent / vr).resolve(),
            (dataset_root / vr).resolve(),
        ]
        val_img = next((c for c in candidates if c.is_dir()), None)

    # 2) 폴백 탐색: A/B 구조 대응
    if train_img is None:
        for p in (
            dataset_root / "train" / "images",
            dataset_root / "images" / "train",
            dataset_root / "data" / "train" / "images",
        ):
            if p.is_dir():
                train_img = p.resolve()
                break
    if val_img is None:
        for p in (
            dataset_root / "val" / "images",
            dataset_root / "valid" / "images",
            dataset_root / "images" / "val",
            dataset_root / "images" / "valid",
            dataset_root / "data" / "val" / "images",
        ):
            if p.is_dir():
                val_img = p.resolve()
                break

    if train_img is None:
        raise RuntimeError(f"train 이미지 디렉터리를 못 찾았습니다: {dataset_root}")

    need_internal_split = False
    if val_img is None:
        logging.warning("val split 을 찾지 못함 — train에서 내부 분할합니다. dataset_root=%s", dataset_root)
        val_img = train_img
        need_internal_split = True

    train_lbl = infer_label_dir_from_image_dir(train_img)
    val_lbl = infer_label_dir_from_image_dir(val_img)

    # 라벨 디렉터리 폴백
    if not train_lbl.is_dir():
        alt = dataset_root / "labels" / "train"
        if alt.is_dir():
            train_lbl = alt.resolve()
    if not val_lbl.is_dir():
        alt = dataset_root / "labels" / "val"
        if alt.is_dir():
            val_lbl = alt.resolve()

    return SplitDirs(
        train_images=train_img,
        train_labels=train_lbl,
        val_images=val_img,
        val_labels=val_lbl,
        need_internal_split=need_internal_split,
    )


def rewrite_label_file(
    src_label: Path,
    dst_label: Path,
    remap: Dict[int, Optional[int]],
    stats: Stats,
    dry_run: bool,
) -> None:
    """
    라벨 파일을 0/1/2만 남기도록 재작성한다.
    """
    text = src_label.read_text(encoding="utf-8", errors="replace").splitlines()
    out_lines: List[str] = []

    for line in text:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 5:
            stats.label_lines_invalid_bbox += 1
            logging.warning("[LABEL] invalid line skipped: %s", src_label)
            continue
        try:
            oid = int(float(parts[0]))
        except ValueError:
            stats.label_lines_invalid_bbox += 1
            continue

        if oid not in remap:
            stats.unknown_class_ids.add(oid)
            stats.label_lines_dropped += 1
            continue

        nid = remap[oid]
        if nid is None:
            stats.label_lines_dropped += 1
            continue

        try:
            xc, yc, w, h = map(float, parts[1:5])
        except ValueError:
            stats.label_lines_invalid_bbox += 1
            continue

        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            stats.label_lines_invalid_bbox += 1
            continue

        out_lines.append(f"{nid} {xc} {yc} {w} {h}\n")

    if not out_lines:
        stats.empty_label_files += 1

    if not dry_run:
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text("".join(out_lines), encoding="utf-8")

    stats.labels_rewritten += 1
    stats.label_lines_kept += len(out_lines)


def copy_images_from_paths(
    split: str,
    image_paths: Iterable[Path],
    src_labels_dir: Path,
    dest_images_dir: Path,
    dest_labels_dir: Path,
    remap: Dict[int, Optional[int]],
    file_prefix: str,
    dry_run: bool,
    stats: Stats,
    used_filenames: Set[str],
) -> None:
    for img in image_paths:
        stem = img.stem
        ext = img.suffix
        dest_filename = f"{file_prefix}{stem}{ext}"
        dest_filename_l = dest_filename.lower()
        if dest_filename_l in used_filenames:
            # prefix는 dataset 단위라 충돌 가능성이 매우 낮지만, 혹시 대비해 뒤에 숫자 붙인다.
            i = 2
            while f"{file_prefix}{stem}_{i}{ext}".lower() in used_filenames:
                i += 1
            dest_filename = f"{file_prefix}{stem}_{i}{ext}"
            dest_filename_l = dest_filename.lower()
        used_filenames.add(dest_filename_l)

        dst_img = dest_images_dir / dest_filename
        dst_label = dest_labels_dir / f"{Path(dest_filename).stem}.txt"

        src_label = src_labels_dir / f"{stem}.txt"
        if not src_label.is_file():
            stats.skipped_missing_label += 1
            stats.warnings.append(f"[{split}] label 없음 => 빈 라벨 생성: {src_label}")
            if not dry_run:
                dest_images_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, dst_img)
                dst_label.parent.mkdir(parents=True, exist_ok=True)
                dst_label.write_text("", encoding="utf-8")
            if split == "train":
                stats.train_images_copied += 1
            else:
                stats.val_images_copied += 1
            continue

        if not dry_run:
            dest_images_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst_img)

        rewrite_label_file(src_label, dst_label, remap, stats, dry_run=dry_run)

        if split == "train":
            stats.train_images_copied += 1
        else:
            stats.val_images_copied += 1


def write_final_data_yaml(dest_root: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    content = (
        f"path: {dest_root}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: fire\n"
        "  1: smoke\n"
        "  2: knife\n"
    )
    (dest_root / "data.yaml").write_text(content, encoding="utf-8")


def clean_dest(dest_root: Path) -> None:
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        p = dest_root / sub
        if p.is_dir():
            shutil.rmtree(p)
    for f in (dest_root / "data.yaml",):
        if f.is_file():
            f.unlink()


def validate_final(dest_root: Path) -> None:
    err: List[str] = []
    d_yaml = dest_root / "data.yaml"
    if not d_yaml.is_file():
        err.append("data.yaml 없음")
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        if not (dest_root / sub).is_dir():
            err.append(f"없음: {sub}")

    img_ext = IMAGE_EXTS
    tr_n = sum(1 for f in (dest_root / "images/train").glob("*") if f.is_file() and f.suffix.lower() in img_ext)
    va_n = sum(1 for f in (dest_root / "images/val").glob("*") if f.is_file() and f.suffix.lower() in img_ext)
    if tr_n == 0:
        err.append("train 이미지 0장")
    if va_n == 0:
        err.append("val 이미지 0장")

    # class id 검증: 0/1/2만 존재해야 함
    for split in ("train", "val"):
        lbl_dir = dest_root / "labels" / split
        if not lbl_dir.is_dir():
            continue
        for lf in lbl_dir.glob("*.txt"):
            txt = lf.read_text(encoding="utf-8", errors="replace")
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    err.append(f"손상된 bbox 줄: {lf}: {line!r}")
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    err.append(f"잘못된 cid: {lf}: {line!r}")
                    continue
                if cid not in (0, 1, 2):
                    err.append(f"class id 0/1/2 아님: {lf}: {cid}")

    if err:
        raise RuntimeError("최종 검증 실패:\n  - " + "\n  - ".join(err))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip-dfire", type=Path, default=DEFAULT_ZIP_DFIRE)
    ap.add_argument("--zip-dfs", type=Path, default=DEFAULT_ZIP_DFS)
    ap.add_argument("--zip-knife", type=Path, default=DEFAULT_ZIP_KNIFE)
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--clean-detect", action="store_true", help="detect_fs_knife 초기화")
    ap.add_argument("--dry-run", action="store_true", help="복사/쓰기 하지 않고 구조만 확인")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="val 폴더 없을 때 train에서 나눌 비율(기본 0.1)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not args.dry_run and args.clean_detect:
        clean_dest(DEST_3CLASS)

    print(f"[INFO] source zip paths: dfire={args.zip_dfire} dfs={args.zip_dfs} knife={args.zip_knife}")

    extract_zip(args.zip_dfire, EXTRACT_DFIRE, force_extract=args.force_extract)
    extract_zip(args.zip_dfs, EXTRACT_DFS, force_extract=args.force_extract)
    extract_zip(args.zip_knife, EXTRACT_KNIFE, force_extract=args.force_extract)

    root_dfire = find_dataset_root(EXTRACT_DFIRE)
    root_dfs = find_dataset_root(EXTRACT_DFS)
    root_knife = find_dataset_root(EXTRACT_KNIFE)

    yaml_dfire = find_yaml_file(root_dfire)
    yaml_dfs = find_yaml_file(root_dfs)
    yaml_knife = find_yaml_file(root_knife)
    if not yaml_dfire or not yaml_dfs or not yaml_knife:
        logging.error("yaml 찾기 실패: dfire=%s dfs=%s knife=%s", yaml_dfire, yaml_dfs, yaml_knife)
        sys.exit(1)

    names_dfire = load_class_names(yaml_dfire)
    names_dfs = load_class_names(yaml_dfs)
    names_knife = load_class_names(yaml_knife)

    remap_dfire = build_remap(names_dfire)
    remap_dfs = build_remap(names_dfs)
    remap_knife = build_remap(names_knife)

    print(f"[INFO] extracted roots: dfire={root_dfire} dfs={root_dfs} knife={root_knife}")
    print(f"[INFO] found yaml paths: dfire={yaml_dfire} dfs={yaml_dfs} knife={yaml_knife}")
    print(f"[INFO] DFire classes: {sorted(names_dfire.values())}")
    print(f"[INFO] DFS classes: {sorted(names_dfs.values())}")
    print(f"[INFO] Knife classes: {sorted(names_knife.values())}")
    print(f"[INFO] DFire remap: {remap_dfire}")
    print(f"[INFO] DFS remap: {remap_dfs}")
    print(f"[INFO] Knife remap: {remap_knife}")

    splits_dfire = discover_split_dirs(root_dfire, yaml_dfire)
    splits_dfs = discover_split_dirs(root_dfs, yaml_dfs)
    splits_knife = discover_split_dirs(root_knife, yaml_knife)

    dest = DEST_3CLASS
    if not args.dry_run:
        (dest / "images/train").mkdir(parents=True, exist_ok=True)
        (dest / "images/val").mkdir(parents=True, exist_ok=True)
        (dest / "labels/train").mkdir(parents=True, exist_ok=True)
        (dest / "labels/val").mkdir(parents=True, exist_ok=True)

    dest_images_train = dest / "images" / "train"
    dest_images_val = dest / "images" / "val"
    dest_labels_train = dest / "labels" / "train"
    dest_labels_val = dest / "labels" / "val"

    used_names: Set[str] = set()
    stats = Stats()

    # --- DFire 복사 ---
    if splits_dfire.need_internal_split:
        all_paths = list(iter_images(splits_dfire.train_images))
        tr_paths, va_paths = partition_train_val(all_paths, args.val_ratio)
        copy_images_from_paths(
            "train",
            tr_paths,
            splits_dfire.train_labels,
            dest_images_train,
            dest_labels_train,
            remap_dfire,
            "dfire_",
            args.dry_run,
            stats,
            used_names,
        )
        copy_images_from_paths(
            "val",
            va_paths,
            splits_dfire.train_labels,
            dest_images_val,
            dest_labels_val,
            remap_dfire,
            "dfire_",
            args.dry_run,
            stats,
            used_names,
        )
    else:
        copy_images_from_paths(
            "train",
            iter_images(splits_dfire.train_images),
            splits_dfire.train_labels,
            dest_images_train,
            dest_labels_train,
            remap_dfire,
            "dfire_",
            args.dry_run,
            stats,
            used_names,
        )
        copy_images_from_paths(
            "val",
            iter_images(splits_dfire.val_images),
            splits_dfire.val_labels,
            dest_images_val,
            dest_labels_val,
            remap_dfire,
            "dfire_",
            args.dry_run,
            stats,
            used_names,
        )

    # --- DFS 복사 (val 없음일 수 있음) ---
    if splits_dfs.need_internal_split:
        all_paths = list(iter_images(splits_dfs.train_images))
        tr_paths, va_paths = partition_train_val(all_paths, args.val_ratio)
        copy_images_from_paths(
            "train",
            tr_paths,
            splits_dfs.train_labels,
            dest_images_train,
            dest_labels_train,
            remap_dfs,
            "dfs_",
            args.dry_run,
            stats,
            used_names,
        )
        copy_images_from_paths(
            "val",
            va_paths,
            splits_dfs.train_labels,
            dest_images_val,
            dest_labels_val,
            remap_dfs,
            "dfs_",
            args.dry_run,
            stats,
            used_names,
        )
    else:
        copy_images_from_paths(
            "train",
            iter_images(splits_dfs.train_images),
            splits_dfs.train_labels,
            dest_images_train,
            dest_labels_train,
            remap_dfs,
            "dfs_",
            args.dry_run,
            stats,
            used_names,
        )
        copy_images_from_paths(
            "val",
            iter_images(splits_dfs.val_images),
            splits_dfs.val_labels,
            dest_images_val,
            dest_labels_val,
            remap_dfs,
            "dfs_",
            args.dry_run,
            stats,
            used_names,
        )

    # --- Knife 복사 ---
    copy_images_from_paths(
        "train",
        iter_images(splits_knife.train_images),
        splits_knife.train_labels,
        dest_images_train,
        dest_labels_train,
        remap_knife,
        "knife_",
        args.dry_run,
        stats,
        used_names,
    )
    if splits_knife.need_internal_split:
        all_paths = list(iter_images(splits_knife.train_images))
        _, va_paths = partition_train_val(all_paths, args.val_ratio)
        copy_images_from_paths(
            "val",
            va_paths,
            splits_knife.train_labels,
            dest_images_val,
            dest_labels_val,
            remap_knife,
            "knife_",
            args.dry_run,
            stats,
            used_names,
        )
    else:
        copy_images_from_paths(
            "val",
            iter_images(splits_knife.val_images),
            splits_knife.val_labels,
            dest_images_val,
            dest_labels_val,
            remap_knife,
            "knife_",
            args.dry_run,
            stats,
            used_names,
        )

    if not args.dry_run:
        write_final_data_yaml(dest)
        validate_final(dest)

    # --- 요약 출력 ---
    print("\n========== 3-class 통합 요약 ==========")
    print(f"train images copied: {stats.train_images_copied}")
    print(f"val images copied:   {stats.val_images_copied}")
    print(f"labels rewritten:    {stats.labels_rewritten}")
    print(f"label lines kept:    {stats.label_lines_kept}")
    print(f"dropped lines:       {stats.label_lines_dropped}")
    print(f"invalid bbox lines:  {stats.label_lines_invalid_bbox}")
    print(f"empty label files:   {stats.empty_label_files}")
    if stats.unknown_class_ids:
        print(f"unknown class ids:   {sorted(stats.unknown_class_ids)}")
    print(f"missing label -> empty txt: {stats.skipped_missing_label}")
    print(f"final dataset root:  {dest}")
    print("=========================================\n")


if __name__ == "__main__":
    main()

