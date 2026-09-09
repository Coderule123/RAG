"""检索用查询清洗：句首语气词、按车型 tag 给检索词加车名前缀。"""

from typing import Dict, List, Optional, Sequence, Tuple

# 长词在前，避免「嗯」抢先匹配「嗯嗯」
_LEADING_FILLERS: Tuple[str, ...] = (
    "嗯嗯",
    "呃呃",
    "嗯呐",
    "嗯啊",
    "嗯",
    "呃",
    "额",
    "啊",
    "哦",
    "噢",
    "唔",
    "唉",
    "嘿",
    "诶",
    "欸",
)

_LEADING_FILLER_TRAIL = "，。、！？,.!?;；…·—-~～、 \t\r\n　"


def strip_leading_fillers(query: str) -> str:
    """去掉句首连续语气词及其后紧跟的标点/空白；剥空则退回原文。"""
    original = query if query is not None else ""
    text = original.strip()
    if not text:
        return original

    while text:
        matched = False
        for filler in _LEADING_FILLERS:
            if text.startswith(filler):
                text = text[len(filler) :].lstrip(_LEADING_FILLER_TRAIL)
                matched = True
                break
        if not matched:
            break

    return text if text else original.strip()


_SKIP_PREFIX_TAGS = frozenset({"general"})
_SKIP_PREFIX_TAG_PREFIXES = ("active_ask",)


def prefix_vehicle_names_for_retrieval(
    query: str,
    tags: Optional[Sequence[str]] = None,
    display_names: Optional[Dict[str, str]] = None,
) -> str:
    """
    在检索词最前按车型 tag 加上「智己LS9」等对外车名，用中文逗号分隔。

    只用于向量检索：general、active_ask* 不加；未知 tag 不加。
    无车型 tag 或 query 为空时返回原文。
    """
    text = (query or "").strip()
    if not text:
        return text

    names: List[str] = []
    seen: set[str] = set()
    mapping = display_names or {}
    for tag in tags or []:
        normalized = (tag or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        if normalized in _SKIP_PREFIX_TAGS:
            continue
        if any(normalized.startswith(prefix) for prefix in _SKIP_PREFIX_TAG_PREFIXES):
            continue
        display = (mapping.get(normalized) or "").strip()
        if not display:
            continue
        seen.add(normalized)
        names.append(display)

    if not names:
        return text
    return "，".join(names + [text])
