import re
from typing import List

from RAG.config.logger_runtime import get_logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = get_logger("rag")


def normalize_for_dedup(text: str) -> str:
    """用于精确去重的轻量归一化：忽略空白与大小写差异。"""
    normalized = re.sub(r"\s+", "", text or "").lower()
    return normalized.strip()


def build_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    """构造按中英文标点与换行优先切分的递归切分器。"""
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


def split_documents(
    docs: List[Document], chunk_size: int, overlap: int
) -> List[Document]:
    """
    将整篇文档切为多个 chunk；metadata 中写入 doc_id / chunk_id 供展示与 sqlite 记录。
    """
    logger.info(
        "开始语义切分: docs=%s chunk_size=%s overlap=%s", len(docs), chunk_size, overlap
    )
    splitter = build_splitter(chunk_size, overlap)
    chunks: List[Document] = []
    seen_docs = set()
    seen_chunks = set()
    skipped_duplicate_docs = 0
    skipped_duplicate_chunks = 0
    for doc_idx, doc in enumerate(docs):
        doc_key = normalize_for_dedup(doc.page_content)
        if not doc_key:
            continue
        if doc_key in seen_docs:
            skipped_duplicate_docs += 1
            continue
        seen_docs.add(doc_key)
        parts = splitter.split_text(doc.page_content)
        for chunk_idx, part in enumerate(parts):
            normalized = normalize_for_dedup(part)
            if not normalized:
                continue
            if normalized in seen_chunks:
                skipped_duplicate_chunks += 1
                continue
            seen_chunks.add(normalized)
            meta = {**doc.metadata, "doc_id": doc_idx, "chunk_id": chunk_idx}
            chunks.append(Document(page_content=part, metadata=meta))
    logger.info(
        "语义切分完成: chunks=%s skipped_duplicate_docs=%s skipped_duplicate_chunks=%s",
        len(chunks),
        skipped_duplicate_docs,
        skipped_duplicate_chunks,
    )
    return chunks
