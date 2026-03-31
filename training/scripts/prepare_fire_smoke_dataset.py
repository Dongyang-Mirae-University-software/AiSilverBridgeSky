#!/usr/bin/env python3
"""
archive.zip + DFS-Fire.yolo26.zip → fire/smoke 2-class 통합 YOLO 데이터셋.

- 압축: training/datasets/_sources/dfire, dfs_fire
- 출력: training/datasets/detect (images/train|val, labels/train|val)
- 파일명: dfire_<원본stem>.<ext>, dfs_<원본stem>.<ext> (prefix로 충돌 방지)
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

# --- 경로 상수 ---
ROOT = Path("/home/apps/SilverBridgeSky/AiSilverBridgeSky")
DATASETS = ROOT / "training" / "datasets"
DEFAULT_ZIP_DFIRE = DATASETS / "archive.zip"
DEFAULT_ZIP_DFS = DATASETS / "DFS-Fire.yolo26.zip"
SOURCES_DIR = DATASETS / "_sources"
EXTRACT_DFIRE = SOURCES_DIR / "dfire"
EXTRACT_DFS = SOURCES_DIR / "dfs_fire"
DEST_DETECT = DATASETS / "detect"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FINAL_CLASSES: Dict[str, int] = {"fire": 0, "smoke": 1}


@dataclass
class SplitDirs:
    """train / val(또는 valid) 이미지·라벨 디렉터리."""

    train_images: Path
    train_labels: Path
    val_images: Path
    val_labels: Path
    # val 폴더가 없고 train 만 있을 때 True → train 목록을 비율로 나눔
    need_internal_split: bool = False


@dataclass
class SummaryStats:
    train_images_copied: int = 0
    val_images_copied: int = 0
    labels_rewritten: int = 0
    label_lines_kept: int = 0
    label_lines_skipped: int = 0
    label_lines_dropped_class: int = 0
    invalid_bbox_lines: int = 0
    empty_label_files: int = 0
    unknown_class_ids: Set[int] = field(default_factory=set)
    warnings: List[str] = field(default_factory=list)


def extract_zip(zip_path: Path, dest_dir: Path, *, force: bool) -> None:
    """zip 압축 해제. dest 에 이미 data.yaml 등이 있고 force 가 아니면 스킵."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"zip 파일 없음: {zip_path}")
    marker = dest_dir / ".extracted_ok"
    if dest_dir.is_dir() and any(dest_dir.iterdir()) and not force:
        if marker.is_file() or find_yaml_file(dest_dir) or _has_any_split(dest_dir):
            logging.info("[SKIP] 이미 압축 해제됨 ( --force-extract 로 재실행 ): %s", dest_dir)
            return
    dest_dir.mkdir(parents=True, exist_ok=True)
    logging.info("압축 해제: %s -> %s", zip_path, dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    marker.write_text("ok\n", encoding="utf-8")


def _has_any_split(root: Path) -> bool:
    for pattern in (
        "train/images",
        "images/train",
        "data/train",
        "valid/images",
        "val/images",
        "images/val",
    ):
        if (root / pattern).is_dir():
            return True
    return False


def find_dataset_root(start_dir: Path) -> Path:
    """압축 루트 안에서 data.yaml 이 있는 디렉터리를 찾는다."""
    start_dir = start_dir.resolve()
    if find_yaml_file(start_dir):
        return start_dir
    subs = [p for p in start_dir.iterdir() if p.is_dir()]
    if len(subs) == 1:
        inner = subs[0]
        if find_yaml_file(inner):
            return inner
        return find_dataset_root(inner)
    for yml in start_dir.rglob("data.yaml"):
        if "site-packages" not in str(yml):
            return yml.parent
    for yml in start_dir.rglob("dataset.yaml"):
        return yml.parent
    return start_dir


def find_yaml_file(dataset_root: Path) -> Optional[Path]:
    for name in ("data.yaml", "dataset.yaml"):
        p = dataset_root / name
        if p.is_file():
            return p
    return None


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
    raise ValueError(f"names 형식 오류: {type(names)}")


def normalize_class_name(name: str) -> Optional[str]:
    """
    최종 카테고리 'fire', 'smoke', 'other', 또는 None(불명·제거).
    """
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 _-]", "", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s in ("smokes", "smoking"):
        s = "smoke"
    if s in ("fires", "flame", "flames"):
        s = "fire"
    if "smoke" in s:
        return "smoke"
    if "fire" in s or s == "flame":
        return "fire"
    if s == "other" or s.startswith("other"):
        return "other"
    return None


def build_class_remap(class_names: Dict[int, str]) -> Dict[int, Optional[int]]:
    """원본 class id -> 최종 0/1 또는 None(제거)."""
    remap: Dict[int, Optional[int]] = {}
    for oid, raw in class_names.items():
        cat = normalize_class_name(raw)
        if cat == "fire":
            remap[oid] = 0
        elif cat == "smoke":
            remap[oid] = 1
        elif cat == "other":
            remap[oid] = None
        else:
            remap[oid] = None
    return remap


def _resolve_yaml_base(yaml_path: Path, raw: Dict[str, Any]) -> Path:
    path_field = raw.get("path")
    if path_field is None or str(path_field).strip() in ("", "."):
        return yaml_path.parent.resolve()
    p = Path(str(path_field))
    if p.is_absolute() and p.is_dir():
        return p.resolve()
    # Kaggle 등 잘못된 절대경로 → yaml 기준 상대 무시
    if p.is_absolute() and not p.is_dir():
        return yaml_path.parent.resolve()
    return (yaml_path.parent / p).resolve()


def _try_dir(*candidates: Path) -> Optional[Path]:
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    return None


def infer_label_dir(image_dir: Path) -> Path:
    """.../images/train -> .../labels/train 등."""
    parts = image_dir.parts
    if "images" in parts:
        idx = list(parts).index("images")
        new_parts = list(parts)
        new_parts[idx] = "labels"
        return Path(*new_parts)
    return image_dir.parent / "labels"


def discover_split_dirs(dataset_root: Path, yaml_path: Path) -> SplitDirs:
    """yaml + 디렉터리 폴백으로 train/val 이미지·라벨 경로 확정."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base = _resolve_yaml_base(yaml_path, raw)
    train_rel = raw.get("train") or raw.get("train_path")
    val_rel = raw.get("val") or raw.get("valid") or raw.get("validation")

    train_img: Optional[Path] = None
    val_img: Optional[Path] = None

    if train_rel:
        tr = str(train_rel).replace("\\", "/").strip()
        # Roboflow 등 잘못된 상대경로보다 일반 폴더명을 우선
        train_img = _try_dir(
            dataset_root / "train" / "images",
            dataset_root / "images" / "train",
            dataset_root / "data" / "train" / "images",
            (base / tr).resolve(),
            (yaml_path.parent / tr).resolve(),
            (dataset_root / tr).resolve(),
        )
        if train_img is None and tr.startswith("../"):
            train_img = _try_dir((yaml_path.parent / tr).resolve())
        if train_img is None:
            alt = tr.lstrip("./").lstrip("../")
            train_img = _try_dir(dataset_root / alt)

    if val_rel:
        vr = str(val_rel).replace("\\", "/").strip()
        val_img = _try_dir(
            dataset_root / "valid" / "images",
            dataset_root / "val" / "images",
            dataset_root / "images" / "val",
            dataset_root / "images" / "valid",
            dataset_root / "data" / "val" / "images",
            (base / vr).resolve(),
            (yaml_path.parent / vr).resolve(),
            (dataset_root / vr).resolve(),
        )
        if val_img is None and vr.startswith("../"):
            val_img = _try_dir((yaml_path.parent / vr).resolve())
        if val_img is None:
            alt = vr.lstrip("./").lstrip("../")
            val_img = _try_dir(dataset_root / alt)

    # 폴백: 구조 A / B
    need_internal_split = False
    if train_img is None:
        train_img = _try_dir(
            dataset_root / "train" / "images",
            dataset_root / "images" / "train",
            dataset_root / "data" / "train" / "images",
        )
    if val_img is None:
        val_img = _try_dir(
            dataset_root / "valid" / "images",
            dataset_root / "val" / "images",
            dataset_root / "images" / "val",
            dataset_root / "images" / "valid",
            dataset_root / "data" / "val" / "images",
        )

    if train_img is None:
        raise RuntimeError(
            f"train 이미지 경로를 찾을 수 없습니다. dataset_root={dataset_root}\n"
            f"  train_rel={train_rel!r} val_rel={val_rel!r}"
        )
    if val_img is None:
        logging.warning(
            "val split 을 찾지 못함 — train 과 동일 경로에서 약 10%% 를 val 로 분할합니다."
        )
        val_img = train_img
        need_internal_split = True

    train_lbl = infer_label_dir(train_img)
    val_lbl = infer_label_dir(val_img)
    if not train_lbl.is_dir():
        alt = _try_dir(dataset_root / "train" / "labels", dataset_root / "data" / "train" / "labels")
        if alt:
            train_lbl = alt
    if not val_lbl.is_dir():
        alt = _try_dir(
            dataset_root / "valid" / "labels",
            dataset_root / "val" / "labels",
            dataset_root / "data" / "val" / "labels",
        )
        if alt:
            val_lbl = alt
    if need_internal_split:
        val_lbl = train_lbl

    logging.info("탐지 train images: %s", train_img)
    logging.info("탐지 train labels: %s", train_lbl)
    logging.info("탐지 val images:   %s", val_img)
    logging.info("탐지 val labels:   %s", val_lbl)
    logging.info("need_internal_split: %s", need_internal_split)

    return SplitDirs(train_img, train_lbl, val_img, val_lbl, need_internal_split)


def rewrite_label_file(
    src_label: Path,
    dst_label: Path,
    remap: Dict[int, Optional[int]],
    stats: SummaryStats,
) -> None:
    """YOLO 라벨을 0/1만 남기도록 재작성."""
    text = src_label.read_text(encoding="utf-8", errors="replace").splitlines()
    out: List[str] = []
    for line in text:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 5:
            stats.invalid_bbox_lines += 1
            logging.warning("bbox 줄 스킵(%s): %s", src_label.name, line[:80])
            continue
        try:
            oid = int(float(parts[0]))
        except ValueError:
            stats.invalid_bbox_lines += 1
            continue
        if oid not in remap:
            stats.unknown_class_ids.add(oid)
            stats.label_lines_dropped_class += 1
            continue
        nid = remap[oid]
        if nid is None:
            stats.label_lines_dropped_class += 1
            continue
        try:
            xc, yc, w, h = map(float, parts[1:5])
        except ValueError:
            stats.invalid_bbox_lines += 1
            continue
        bbox_ok = all(0.0 <= v <= 1.0 for v in (xc, yc, w, h))
        if not bbox_ok:
            stats.invalid_bbox_lines += 1
            logging.warning("정규화 좌표 범위 밖(%s): %s", src_label.name, line[:80])
            continue
        parts[0] = str(nid)
        out.append(" ".join(parts) + "\n")
        stats.label_lines_kept += 1

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("".join(out), encoding="utf-8")
    if not out:
        stats.empty_label_files += 1


def iter_images(folder: Path) -> Iterator[Path]:
    if not folder.is_dir():
        return
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def partition_train_val(paths: List[Path], val_ratio: float = 0.1) -> Tuple[List[Path], List[Path]]:
    """val 폴더가 없을 때 train 목록을 train/val 로 분할."""
    paths = sorted(paths, key=lambda p: p.name)
    n = len(paths)
    if n == 0:
        return [], []
    if n == 1:
        return paths, paths
    val_n = max(1, int(n * val_ratio))
    val_n = min(val_n, n - 1)
    return paths[:-val_n], paths[-val_n:]


def copy_images_from_paths(
    split: str,
    image_paths: Iterable[Path],
    src_labels: Path,
    dest_images: Path,
    dest_labels: Path,
    remap: Dict[int, Optional[int]],
    file_prefix: str,
    dry_run: bool,
    stats: SummaryStats,
    used_names: Set[str],
) -> None:
    """이미지 경로 목록 기준 복사 + 라벨 재작성."""
    for img in image_paths:
        stem = img.stem
        ext = img.suffix
        src_txt = src_labels / f"{stem}.txt"
        dest_base = f"{file_prefix}{stem}{ext}"
        if dest_base.lower() in used_names:
            n = 2
            while f"{file_prefix}{stem}_{n}{ext}".lower() in used_names:
                n += 1
            dest_base = f"{file_prefix}{stem}_{n}{ext}"
        used_names.add(dest_base.lower())

        dest_img = dest_images / dest_base
        dest_lbl = dest_labels / (Path(dest_base).stem + ".txt")

        if not src_txt.is_file():
            stats.warnings.append(f"[{split}] 라벨 없음 → 빈 라벨: {file_prefix}{stem}{ext}")
            if not dry_run:
                dest_images.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, dest_img)
                dest_labels.mkdir(parents=True, exist_ok=True)
                dest_lbl.write_text("", encoding="utf-8")
                stats.empty_label_files += 1
            if split == "train":
                stats.train_images_copied += 1
            else:
                stats.val_images_copied += 1
            stats.labels_rewritten += 1
            continue

        if not dry_run:
            dest_images.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest_img)
            rewrite_label_file(src_txt, dest_lbl, remap, stats)
        stats.labels_rewritten += 1
        if split == "train":
            stats.train_images_copied += 1
        else:
            stats.val_images_copied += 1


def copy_split_images_and_labels(
    split: str,
    src_images: Path,
    src_labels: Path,
    dest_images: Path,
    dest_labels: Path,
    remap: Dict[int, Optional[int]],
    file_prefix: str,
    dry_run: bool,
    stats: SummaryStats,
    used_names: Set[str],
) -> None:
    copy_images_from_paths(
        split,
        iter_images(src_images),
        src_labels,
        dest_images,
        dest_labels,
        remap,
        file_prefix,
        dry_run,
        stats,
        used_names,
    )


def write_final_data_yaml(dest_root: Path) -> None:
    content = (
        f"path: {dest_root}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: fire\n"
        "  1: smoke\n"
    )
    dest_root.mkdir(parents=True, exist_ok=True)
    (dest_root / "data.yaml").write_text(content, encoding="utf-8")
    logging.info("작성: %s", dest_root / "data.yaml")


def clean_detect_root(dest: Path) -> None:
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        p = dest / sub
        if p.is_dir():
            shutil.rmtree(p)
            logging.info("삭제: %s", p)
    for f in (dest / "data.yaml",):
        if f.is_file():
            f.unlink()


def validate_final(dest: Path) -> None:
    err: List[str] = []
    if not (dest / "data.yaml").is_file():
        err.append("data.yaml 없음")
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        if not (dest / sub).is_dir():
            err.append(f"없음: {sub}")
    img_ext = IMAGE_EXTS
    nt = sum(1 for f in (dest / "images/train").glob("*") if f.suffix.lower() in img_ext)
    nv = sum(1 for f in (dest / "images/val").glob("*") if f.suffix.lower() in img_ext)
    if nt == 0:
        err.append("train 이미지 0장")
    if nv == 0:
        err.append("val 이미지 0장")

    for split in ("train", "val"):
        for lf in (dest / "labels" / split).glob("*.txt"):
            for line in lf.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    err.append(f"손상 라벨 {lf}")
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    err.append(f"잘못된 class {lf}")
                    continue
                if cid not in (0, 1):
                    err.append(f"class id 0,1 아님: {lf} -> {cid}")

    if err:
        raise RuntimeError("최종 검증 실패:\n  - " + "\n  - ".join(err))


def print_final_summary(
    zip_dfire: Path,
    zip_dfs: Path,
    roots: Dict[str, Path],
    yamls: Dict[str, Path],
    names: Dict[str, Dict[int, str]],
    remaps: Dict[str, Dict[int, Optional[int]]],
    dest: Path,
    stats: SummaryStats,
) -> None:
    print("\n========== 통합 데이터셋 요약 ==========")
    print(f"source zip DFire: {zip_dfire}")
    print(f"source zip DFS:   {zip_dfs}")
    for k, r in roots.items():
        print(f"extracted root [{k}]: {r}")
    for k, y in yamls.items():
        print(f"yaml [{k}]: {y}")
    for k, n in names.items():
        print(f"[INFO] {k} original class names: {n}")
    for k, m in remaps.items():
        print(f"[INFO] {k} remap (orig_id -> final or None): {m}")
    print(f"[INFO] train images copied: {stats.train_images_copied}")
    print(f"[INFO] val images copied:   {stats.val_images_copied}")
    print(f"[INFO] labels rewritten:    {stats.labels_rewritten}")
    print(f"[INFO] label lines kept:    {stats.label_lines_kept}")
    print(f"[INFO] lines dropped (class): {stats.label_lines_dropped_class}")
    print(f"[INFO] invalid bbox lines:  {stats.invalid_bbox_lines}")
    print(f"[INFO] empty label files:   {stats.empty_label_files}")
    if stats.unknown_class_ids:
        print(f"[INFO] unknown class ids seen: {sorted(stats.unknown_class_ids)}")
    print(f"[INFO] final dataset root: {dest}")
    print("========================================\n")


def process_one_source(
    name: str,
    dataset_root: Path,
    file_prefix: str,
    dest: Path,
    dry_run: bool,
    stats: SummaryStats,
    used_train: Set[str],
    used_val: Set[str],
) -> Tuple[Dict[int, str], Dict[int, Optional[int]]]:
    yaml_path = find_yaml_file(dataset_root)
    if not yaml_path:
        raise FileNotFoundError(f"{name}: data.yaml 없음 — {dataset_root}")
    class_names = load_class_names(yaml_path)
    if not class_names:
        raise ValueError(f"{name}: names 비어 있음 — {yaml_path}")
    remap = build_class_remap(class_names)
    splits = discover_split_dirs(dataset_root, yaml_path)

    if splits.need_internal_split:
        all_paths = list(iter_images(splits.train_images))
        tr_paths, va_paths = partition_train_val(all_paths, 0.1)
        copy_images_from_paths(
            "train",
            tr_paths,
            splits.train_labels,
            dest / "images" / "train",
            dest / "labels" / "train",
            remap,
            file_prefix,
            dry_run,
            stats,
            used_train,
        )
        copy_images_from_paths(
            "val",
            va_paths,
            splits.train_labels,
            dest / "images" / "val",
            dest / "labels" / "val",
            remap,
            file_prefix,
            dry_run,
            stats,
            used_val,
        )
    else:
        copy_split_images_and_labels(
            "train",
            splits.train_images,
            splits.train_labels,
            dest / "images" / "train",
            dest / "labels" / "train",
            remap,
            file_prefix,
            dry_run,
            stats,
            used_train,
        )
        copy_split_images_and_labels(
            "val",
            splits.val_images,
            splits.val_labels,
            dest / "images" / "val",
            dest / "labels" / "val",
            remap,
            file_prefix,
            dry_run,
            stats,
            used_val,
        )
    return class_names, remap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip-dfire", type=Path, default=DEFAULT_ZIP_DFIRE)
    p.add_argument("--zip-dfs", type=Path, default=DEFAULT_ZIP_DFS)
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--clean-detect", action="store_true", help="detect/ 하위 초기화 후 통합")
    p.add_argument("--dry-run", action="store_true", help="detect/ 에 쓰지 않고 로그만")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    stats = SummaryStats()
    used_train: Set[str] = set()
    used_val: Set[str] = set()

    if not args.zip_dfire.is_file():
        logging.error("없음: %s", args.zip_dfire)
        sys.exit(1)
    if not args.zip_dfs.is_file():
        logging.error("없음: %s", args.zip_dfs)
        sys.exit(1)

    if args.dry_run:
        logging.info("*** DRY-RUN: detect/ 에 출력하지 않습니다 ***")

    try:
        extract_zip(args.zip_dfire, EXTRACT_DFIRE, force=args.force_extract)
        extract_zip(args.zip_dfs, EXTRACT_DFS, force=args.force_extract)
    except Exception as e:
        logging.error("압축 해제 실패: %s", e)
        sys.exit(1)

    root_dfire = find_dataset_root(EXTRACT_DFIRE)
    root_dfs = find_dataset_root(EXTRACT_DFS)
    yaml_dfire = find_yaml_file(root_dfire)
    yaml_dfs = find_yaml_file(root_dfs)
    if not yaml_dfire:
        logging.error("DFire 쪽 data.yaml 없음: %s", root_dfire)
        sys.exit(1)
    if not yaml_dfs:
        logging.error("DFS 쪽 data.yaml 없음: %s", root_dfs)
        sys.exit(1)

    names: Dict[str, Dict[int, str]] = {}
    remaps: Dict[str, Dict[int, Optional[int]]] = {}

    if args.clean_detect and not args.dry_run:
        clean_detect_root(DEST_DETECT)
    elif args.clean_detect and args.dry_run:
        logging.info("--clean-detect 는 dry-run 과 함께 무시됩니다.")

    dest = DEST_DETECT
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    try:
        n1, r1 = process_one_source(
            "DFire", root_dfire, "dfire_", dest, args.dry_run, stats, used_train, used_val
        )
        names["DFire"] = n1
        remaps["DFire"] = r1
        n2, r2 = process_one_source(
            "DFS", root_dfs, "dfs_", dest, args.dry_run, stats, used_train, used_val
        )
        names["DFS"] = n2
        remaps["DFS"] = r2
    except Exception as e:
        logging.error("통합 실패: %s", e)
        sys.exit(1)

    if not args.dry_run:
        write_final_data_yaml(dest)
        try:
            validate_final(dest)
            logging.info("최종 검증 통과.")
        except Exception as e:
            logging.error("%s", e)
            sys.exit(1)

    print_final_summary(
        args.zip_dfire,
        args.zip_dfs,
        {"dfire": root_dfire, "dfs": root_dfs},
        {"dfire": yaml_dfire, "dfs": yaml_dfs},
        names,
        remaps,
        dest,
        stats,
    )


if __name__ == "__main__":
    main()
