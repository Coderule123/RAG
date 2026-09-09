"""检索用查询清洗：只去掉句首语气词，不剥请求外壳。"""

from typing import Tuple

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
