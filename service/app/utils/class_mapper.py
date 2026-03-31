"""
모델 원본 클래스명 → 화면/API 표시용 이름 정규화.

고정 인덱스(0=DFS 등)는 사용하지 않는다.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# 서비스가 “지원 예정”으로 알고 있는 표시명 (고정 순서, 표시·missing 계산용)
CONFIGURED_DISPLAY_CLASSES: tuple[str, ...] = ("DFS", "D-fire", "Knife", "Fall")


def _norm_key(s: str) -> str:
    """비교용: 소문자, 공백 축소, 일부 특수문자 제거."""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_class_name(raw_name: str) -> str:
    """
    모델에서 온 클래스 문자열을 표시용으로 정규화한다.
    규칙에 맞지 않으면 원본을 그대로 반환한다.
    """
    if raw_name is None:
        return ""
    original = str(raw_name).strip()
    if not original:
        return ""

    k = _norm_key(original)

    # 키워드 우선순위: 더 구체적인 패턴을 먼저
    if "넘어짐" in original or "fall" in k or "fallen" in k:
        return "Fall"
    if "knife" in k:
        return "Knife"
    if "d-fire" in k or "dfire" in k.replace("-", "") or "d_fire" in k:
        return "D-fire"
    if "dfs" in k and "knife" not in k and "fire" not in k:
        # dfs 단독 토큰에 가깝게
        if re.search(r"\bdfs\b", k) or k == "dfs" or k.startswith("dfs"):
            return "DFS"
    if "dfs" in k:
        return "DFS"
    if "fire" in k or "smoke" in k:
        # 학습 데이터에 fire/smoke만 있을 때 D-fire 로 묶어 표시 (요구: fire, d-fire → D-fire)
        return "D-fire"

    return original


def build_model_class_summary(model_names: Any) -> dict[str, Any]:
    """
    모델 클래스 목록(dict/list)을 받아 요약 dict 생성.

    Returns:
        configured_classes, model_classes (원본 순서 유지),
        normalized_model_classes, missing_classes
    """
    raw_list: list[str] = []
    if model_names is None:
        raw_list = []
    elif isinstance(model_names, dict):
        try:
            keys = sorted(int(k) for k in model_names.keys())
            raw_list = [str(model_names[i]) for i in keys]
        except (TypeError, ValueError, KeyError):
            raw_list = [str(v) for v in model_names.values()]
    elif isinstance(model_names, (list, tuple)):
        raw_list = [str(x) for x in model_names]
    else:
        raw_list = [str(model_names)]

    normalized = [normalize_class_name(n) for n in raw_list]
    configured = list(CONFIGURED_DISPLAY_CLASSES)
    norm_set = set(normalized)
    missing = [c for c in configured if c not in norm_set]

    return {
        "configured_classes": configured,
        "model_classes": raw_list,
        "normalized_model_classes": normalized,
        "missing_classes": missing,
    }
