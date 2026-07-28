"""
结构感知 + 语义切分器。

切分策略（按优先级）：
1. QA / JSON 记录（record_type 为 qa_pair / json_item / json_document）保持原子性，
   不再二次切分（除非超长），维持既有 JSON 问答入库效果。
2. 含 Markdown 标题结构的文档（txt/md/pdf/docx 加载层已统一转为 Markdown 标题），
   按标题树切成章节：
   - 小章节按「同一父级路径」合并至接近 chunk_size；
   - 超长章节二次切分（优先 embedding 语义断点，退回递归字符切分）；
   - 每个 chunk 头部注入「文档标题 > 章节路径」上下文（可配置），
     使向量与 prompt 上下文都携带车型/章节信息。
3. 无结构长文本：优先 embedding 语义断点切分（相邻句子余弦相似度低谷处断开），
   无 embedding 时退回递归字符切分。
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from RAG.config.logger_runtime import get_logger

logger = get_logger("rag")

# 加载层（PDF）注入的页码标记，独占一行
PAGE_MARKER_RE = re.compile(r"^\[\[PAGE=(\d+)\]\]\s*$")
# Markdown 标题行
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# 中文/英文句子边界（用于语义断点切分）
_SENTENCE_END_RE = re.compile(r"(?<=[。！？!?；;])\s*")
# 列表项/表格行（页码标记移除后判断是否可与上一行拼接）
_BULLET_RE = re.compile(r"^\s*([-*•·]|\d+[.、）)]|[（(]?[一二三四五六七八九十]+[）)、.])\s+")
# 句末终止符（判断跨页断句）
_TERMINAL_CHARS = "。！？!?；;：:…」》】’”"

# 保持原子性的记录类型（JSON 问答等），不做二次切分
ATOMIC_RECORD_TYPES = {"qa_pair", "json_item", "json_document"}


def normalize_for_dedup(text: str) -> str:
    """用于精确去重的轻量归一化：忽略空白与大小写差异。"""
    normalized = re.sub(r"\s+", "", text or "").lower()
    return normalized.strip()


def build_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    """构造按中英文标点与换行优先切分的递归切分器（作为语义切分的回退方案）。"""
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            ". ",
            "! ",
            "? ",
            "，",
            ",",
            " ",
            "",
        ],
        keep_separator=True,
    )


@dataclass
class _Section:
    """标题树切出的章节：path 为标题路径（不含文档标题），content 为正文。"""

    path: List[str] = field(default_factory=list)
    content: str = ""
    page: Optional[int] = None


def _doc_title(doc: Document) -> str:
    """从 source 文件名推导文档标题：智己LS6_参数配置.txt -> 智己LS6 参数配置。"""
    source = str(doc.metadata.get("source", "") or "")
    if not source:
        return ""
    stem = source.replace("\\", "/").rsplit("/", 1)[-1]
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", stem)
    return stem.replace("_", " ").strip()


def _parse_sections(text: str) -> List[_Section]:
    """
    按 Markdown 标题解析章节树；同时消费 [[PAGE=n]] 页码标记：
    - 记录每个章节起始页；
    - 标记处若上一行未以句末标点结束且下一行不是标题/列表项，则拼接断句（跨页修复）。
    """
    sections: List[_Section] = [_Section()]
    # 各级标题栈：heading_stack[i] 为第 i+1 级标题文本
    heading_stack: List[str] = []
    current_page: Optional[int] = None
    join_next_line = False

    def _current(clone_page: Optional[int]) -> _Section:
        sec = sections[-1]
        if sec.page is None and clone_page is not None:
            sec.page = clone_page
        return sec

    for raw_line in text.split("\n"):
        marker = PAGE_MARKER_RE.match(raw_line)
        if marker:
            current_page = int(marker.group(1))
            sec = sections[-1]
            body = sec.content.rstrip()
            join_next_line = bool(body) and body[-1] not in _TERMINAL_CHARS
            continue

        heading = HEADING_RE.match(raw_line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(title)
            sections.append(
                _Section(path=[h for h in heading_stack if h], page=current_page)
            )
            join_next_line = False
            continue

        line = raw_line.rstrip()
        if not line.strip():
            sec = _current(current_page)
            if sec.content and not sec.content.endswith("\n\n"):
                sec.content += "\n"
            join_next_line = False
            continue

        sec = _current(current_page)
        if (
            join_next_line
            and sec.content.strip()
            and not _BULLET_RE.match(line)
            and not line.lstrip().startswith("|")
        ):
            sec.content = sec.content.rstrip("\n") + line.strip() + "\n"
        else:
            sec.content += line + "\n"
        join_next_line = False

    return [
        _Section(path=s.path, content=s.content.strip(), page=s.page)
        for s in sections
        if s.content.strip() or s.path
    ]


def _merge_sections(sections: List[_Section], chunk_size: int) -> List[_Section]:
    """
    将「同一父级路径」下的连续小章节合并至接近 chunk_size，避免碎片化。
    被合并的子章节标题以「◆ 标题」形式保留在正文中。
    """
    merged: List[_Section] = []
    for sec in sections:
        rendered = sec.content
        if sec.path:
            rendered = f"◆ {sec.path[-1]}\n{sec.content}" if sec.content else f"◆ {sec.path[-1]}"
        parent = sec.path[:-1] if sec.path else []

        if merged:
            prev = merged[-1]
            # prev 已是合并块时 path 即父路径；未合并时父路径为 path[:-1]
            prev_parent = prev.path if getattr(prev, "_is_group", False) else (prev.path[:-1] if prev.path else [])
            if (
                prev_parent == parent
                and len(prev.content) + len(rendered) + 1 <= chunk_size
                and len(prev.content) < chunk_size
            ):
                if not getattr(prev, "_is_group", False):
                    # 将 prev 降级为「父路径分组块」，其自身标题也保留到正文
                    own = f"◆ {prev.path[-1]}\n{prev.content}" if prev.path else prev.content
                    prev.content = own.strip()
                    prev.path = parent
                    prev._is_group = True  # type: ignore[attr-defined]
                prev.content = f"{prev.content}\n{rendered}".strip()
                if prev.page is None:
                    prev.page = sec.page
                continue

        new_sec = _Section(path=list(sec.path), content=sec.content, page=sec.page)
        merged.append(new_sec)
    return merged


def _split_sentences(text: str) -> List[str]:
    """按句末标点与换行切句，保留列表项完整性。"""
    sentences: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _BULLET_RE.match(line) or line.startswith("|") or line.startswith("◆"):
            sentences.append(line)
            continue
        parts = [p.strip() for p in _SENTENCE_END_RE.split(line) if p.strip()]
        sentences.extend(parts if parts else [line])
    return sentences


def _semantic_breakpoints(
    sentences: Sequence[str],
    embed_fn: Callable[[List[str]], List[List[float]]],
    percentile: float,
) -> List[int]:
    """
    计算语义断点：相邻句向量余弦距离超过给定百分位处断开。
    返回断点下标集合（在该句 *之后* 断开）。
    """
    vectors = np.asarray(embed_fn(list(sentences)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    sims = (vectors[:-1] * vectors[1:]).sum(axis=1)
    distances = 1.0 - sims
    threshold = float(np.percentile(distances, percentile))
    return [i for i, d in enumerate(distances) if d >= threshold]


def _group_sentences(
    sentences: List[str],
    breakpoints: Optional[List[int]],
    chunk_size: int,
    max_chunk_size: int,
) -> List[str]:
    """按断点分组后再按长度约束合并/硬切，输出最终文本块。"""
    bp_set = set(breakpoints or [])
    groups: List[str] = []
    buf: List[str] = []
    buf_len = 0
    for idx, sent in enumerate(sentences):
        sep = 1 if buf else 0
        if buf and (buf_len + sep + len(sent) > max_chunk_size):
            groups.append("\n".join(buf))
            buf, buf_len = [], 0
        buf.append(sent)
        buf_len += sep + len(sent)
        if idx in bp_set and buf_len >= chunk_size // 2:
            groups.append("\n".join(buf))
            buf, buf_len = [], 0
    if buf:
        groups.append("\n".join(buf))

    # 合并过小的相邻组，避免碎片
    merged: List[str] = []
    for g in groups:
        if merged and len(merged[-1]) + len(g) + 1 <= chunk_size:
            merged[-1] = f"{merged[-1]}\n{g}"
        else:
            merged.append(g)
    return merged


def _split_long_text(
    text: str,
    chunk_size: int,
    overlap: int,
    max_chunk_size: int,
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]],
    percentile: float,
) -> List[str]:
    """
    超长文本二次切分：优先 embedding 语义断点，失败或句子过少时退回递归字符切分。
    返回 (文本块列表)。
    """
    sentences = _split_sentences(text)
    if embed_fn is not None and len(sentences) >= 6:
        try:
            breakpoints = _semantic_breakpoints(sentences, embed_fn, percentile)
            return _group_sentences(sentences, breakpoints, chunk_size, max_chunk_size)
        except Exception as exc:  # noqa: BLE001
            logger.warning("语义断点切分失败，退回递归切分: %s", exc)
    splitter = build_splitter(chunk_size, overlap)
    return [p for p in splitter.split_text(text) if p.strip()]


def _context_header(title: str, path: List[str]) -> str:
    """构造 chunk 头部的上下文行：【所属章节】文档标题 > 一级标题 > 二级标题。"""
    crumbs: List[str] = []
    if title:
        crumbs.append(title)
    for part in path:
        # 跳过与上一级重复的标题（如文件名与 H1 相同）
        if crumbs and normalize_for_dedup(crumbs[-1]) == normalize_for_dedup(part):
            continue
        crumbs.append(part)
    if not crumbs:
        return ""
    return "【所属章节】" + " > ".join(crumbs)


def split_documents(
    docs: List[Document],
    chunk_size: int,
    overlap: int,
    *,
    max_chunk_size: Optional[int] = None,
    min_chunk_chars: int = 15,
    add_context_header: bool = True,
    embeddings_model=None,
    semantic_split: bool = True,
    semantic_breakpoint_percentile: float = 88.0,
) -> List[Document]:
    """
    将整篇文档切为多个 chunk；metadata 中写入 doc_id / chunk_id / section_path /
    split_method 供检索展示与 sqlite 记录。

    - chunk_size：目标块大小（章节合并预算）
    - max_chunk_size：允许的最大块大小，超出则二次切分（默认 1.4 * chunk_size）
    - add_context_header：是否在 chunk 头部注入「文档标题 > 章节路径」
    - embeddings_model：EmbeddingService 实例；提供且 semantic_split 为真时，
      无结构长文本使用语义断点切分
    """
    if max_chunk_size is None:
        max_chunk_size = int(chunk_size * 1.4)
    max_chunk_size = max(max_chunk_size, chunk_size)

    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None
    if semantic_split and embeddings_model is not None:
        embed_fn = embeddings_model.model.embed_documents

    logger.info(
        "开始切分: docs=%s chunk_size=%s overlap=%s max_chunk_size=%s semantic=%s",
        len(docs),
        chunk_size,
        overlap,
        max_chunk_size,
        embed_fn is not None,
    )

    chunks: List[Document] = []
    seen_docs = set()
    seen_chunks = set()
    skipped_duplicate_docs = 0
    skipped_duplicate_chunks = 0

    def _emit(
        doc: Document,
        doc_idx: int,
        chunk_counter: List[int],
        body: str,
        section_path: List[str],
        page: Optional[int],
        split_method: str,
        title: str,
    ) -> None:
        nonlocal skipped_duplicate_chunks
        body = body.strip()
        if len(body) < min_chunk_chars:
            return
        text = body
        if add_context_header:
            header = _context_header(title, section_path)
            if header:
                text = f"{header}\n{body}"
        normalized = normalize_for_dedup(text)
        if not normalized:
            return
        if normalized in seen_chunks:
            skipped_duplicate_chunks += 1
            return
        seen_chunks.add(normalized)
        meta: Dict = {
            **doc.metadata,
            "doc_id": doc_idx,
            "chunk_id": chunk_counter[0],
            "split_method": split_method,
        }
        if section_path:
            meta["section_path"] = " > ".join(section_path)
        if page is not None:
            meta["page"] = page
        chunks.append(Document(page_content=text, metadata=meta))
        chunk_counter[0] += 1

    for doc_idx, doc in enumerate(docs):
        doc_key = normalize_for_dedup(doc.page_content)
        if not doc_key:
            continue
        if doc_key in seen_docs:
            skipped_duplicate_docs += 1
            continue
        seen_docs.add(doc_key)

        chunk_counter = [0]
        record_type = str(doc.metadata.get("record_type", "") or "")
        title = _doc_title(doc)

        # 1) JSON 问答等原子记录：不加上下文头、不二次切分（超长时按递归切分兜底）
        if record_type in ATOMIC_RECORD_TYPES:
            content = doc.page_content.strip()
            if len(content) <= max_chunk_size:
                normalized = normalize_for_dedup(content)
                if normalized and normalized not in seen_chunks:
                    seen_chunks.add(normalized)
                    chunks.append(
                        Document(
                            page_content=content,
                            metadata={
                                **doc.metadata,
                                "doc_id": doc_idx,
                                "chunk_id": 0,
                                "split_method": "atomic",
                            },
                        )
                    )
                elif normalized:
                    skipped_duplicate_chunks += 1
                continue
            for part in build_splitter(chunk_size, overlap).split_text(content):
                _emit(
                    doc, doc_idx, chunk_counter, part, [], None, "recursive", ""
                )
            continue

        # 2) 结构感知切分：解析标题树 -> 合并小节 -> 超长二次切分
        sections = _parse_sections(doc.page_content)
        sections = _merge_sections(sections, chunk_size)

        for sec in sections:
            body = sec.content.strip()
            if not body:
                continue
            if len(body) <= max_chunk_size:
                _emit(
                    doc,
                    doc_idx,
                    chunk_counter,
                    body,
                    sec.path,
                    sec.page,
                    "structure",
                    title,
                )
                continue
            # 章节内是否还有子结构（列表密集时用递归切分保持条目完整）
            parts = _split_long_text(
                body,
                chunk_size,
                overlap,
                max_chunk_size,
                embed_fn,
                semantic_breakpoint_percentile,
            )
            method = "semantic" if embed_fn is not None and len(_split_sentences(body)) >= 6 else "recursive"
            for part in parts:
                _emit(
                    doc,
                    doc_idx,
                    chunk_counter,
                    part,
                    sec.path,
                    sec.page,
                    method,
                    title,
                )

    logger.info(
        "切分完成: chunks=%s skipped_duplicate_docs=%s skipped_duplicate_chunks=%s",
        len(chunks),
        skipped_duplicate_docs,
        skipped_duplicate_chunks,
    )
    return chunks
