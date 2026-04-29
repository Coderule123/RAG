import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

from RAG.config.logger_runtime import get_logger

logger = get_logger("rag")

# 按扩展名选择 Loader；未列出的后缀会被跳过
LOADER_BY_SUFFIX = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
    ".json": "json",
}


def get_loader(file_path: Path):
    """根据后缀构造对应 Loader；不支持的类型返回 None。"""
    suffix = file_path.suffix.lower()
    loader_cls = LOADER_BY_SUFFIX.get(suffix)
    if not loader_cls:
        return None
    if suffix == ".json":
        return "json"
    if loader_cls is TextLoader:
        return loader_cls(str(file_path), encoding="utf-8", autodetect_encoding=True)
    return loader_cls(str(file_path))


def compute_file_hash(file_path: Path) -> str:
    """计算文件哈希，用于增量模式下判定文件是否变化。"""
    sha = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def load_hash_registry(metadata_path: Optional[str]) -> Dict[str, str]:
    """
    从 metadata.json 提取 source -> doc_hash 映射。
    metadata 兼容旧格式：若缺失 doc_hash，则该文件下次会被重新处理一次。
    """
    if not metadata_path:
        return {}

    path = Path(metadata_path)
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("读取 metadata 失败，将忽略增量缓存: %s, err=%s", path, exc)
        return {}

    if isinstance(payload, dict):
        records = payload.get("chunks", [])
    elif isinstance(payload, list):
        records = payload
    else:
        return {}

    registry: Dict[str, str] = {}
    for item in records:
        metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
        source = metadata.get("source")
        doc_hash = metadata.get("doc_hash")
        if source and doc_hash:
            registry[source] = doc_hash
    return registry


def _compact_fragmented_line(line: str) -> str:
    """修复 PDF 抽取后被拆散的字符序列，如 '4 0 1 . 0'。"""
    tokens = line.split()
    if len(tokens) < 4:
        return line
    single_char_ratio = sum(1 for token in tokens if len(token) == 1) / len(tokens)
    if single_char_ratio < 0.6:
        return line
    return "".join(tokens)


def clean_text(text: str) -> str:
    """清洗 PDF/文本中的控制字符、空字节与明显噪音行。"""
    if not text:
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: List[str] = []
    for raw_line in text.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        line = _compact_fragmented_line(line)
        visible = re.sub(r"\s+", "", line)
        meaningful_chars = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", visible)
        if not meaningful_chars and len(visible) >= 3:
            continue
        if meaningful_chars and (len(meaningful_chars) / max(len(visible), 1)) < 0.25:
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_intent_suffix(text: str) -> str:
    return re.sub(r"\s*<INTENT>.*?</INTENT>\s*$", "", text, flags=re.DOTALL).strip()


def load_json_documents(file_path: Path, file_hash: str) -> List[Document]:
    """
    加载 JSON 文件。
    若是问答数组（如 instruction/input/output），则将每个问答对展开成一条独立 Document。
    """
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    source = str(file_path.resolve())

    if isinstance(payload, list):
        documents: List[Document] = []
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                text = clean_text(json.dumps(item, ensure_ascii=False))
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": source,
                                "page": idx,
                                "doc_hash": file_hash,
                                "record_type": "json_item",
                            },
                        )
                    )
                continue

            instruction = clean_text(str(item.get("instruction", "")).strip())
            user_input = clean_text(str(item.get("input", "")).strip())
            output = _strip_intent_suffix(clean_text(str(item.get("output", "")).strip()))
            system = clean_text(str(item.get("system", "")).strip())

            if instruction or output:
                parts = []
                if system:
                    parts.append(f"系统：{system}")
                if instruction:
                    parts.append(f"问题：{instruction}")
                if user_input:
                    parts.append(f"补充信息：{user_input}")
                if output:
                    parts.append(f"答案：{output}")
                page_content = "\n".join(parts).strip()
                if page_content:
                    documents.append(
                        Document(
                            page_content=page_content,
                            metadata={
                                "source": source,
                                "page": idx,
                                "doc_hash": file_hash,
                                "record_type": "qa_pair",
                                "qa_index": idx,
                            },
                        )
                    )
                continue

            text = clean_text(json.dumps(item, ensure_ascii=False))
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source,
                            "page": idx,
                            "doc_hash": file_hash,
                            "record_type": "json_item",
                        },
                    )
                )
        return documents

    text = clean_text(json.dumps(payload, ensure_ascii=False, indent=2))
    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata={
                "source": source,
                "page": 0,
                "doc_hash": file_hash,
                "record_type": "json_document",
            },
        )
    ]


def load_documents(
    data_dir: str, incremental: bool = True, metadata_path: Optional[str] = None
) -> Tuple[List[Document], Dict[str, int]]:
    """
    递归扫描 data_dir 下支持的文件，加载为 LangChain Document 列表。
    返回 (文档列表, 统计信息)；每条记录补充 source / page / doc_hash 元数据便于追溯。
    增量模式通过 metadata.json 中已有 chunk 的 doc_hash 判定新增/变更文件。
    """
    logger.info("开始加载文档: %s", data_dir)
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    registry = load_hash_registry(metadata_path) if incremental else {}
    records: List[Document] = []
    failed_files = 0
    skipped_unchanged = 0
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        file_hash = compute_file_hash(file_path)
        source = str(file_path.resolve())
        if incremental and registry.get(source) == file_hash:
            skipped_unchanged += 1
            continue
        loader = get_loader(file_path)
        if not loader:
            continue
        try:
            if file_path.suffix.lower() == ".json":
                docs = load_json_documents(file_path, file_hash)
            else:
                docs = loader.load()
            logger.info(f"成功加载文档: {file_path}")
        except Exception as exc:
            failed_files += 1
            logger.exception("加载失败，跳过文件: %s, err=%s", str(file_path), exc)
            continue
        for page_idx, doc in enumerate(docs):
            text = clean_text(doc.page_content or "")
            if not text:
                continue
            metadata = dict(doc.metadata)
            metadata["source"] = metadata.get("source", source)
            metadata["page"] = metadata.get(
                "page", metadata.get("page_number", page_idx)
            )
            metadata["doc_hash"] = metadata.get("doc_hash", file_hash)
            records.append(Document(page_content=text, metadata=metadata))
    stats = {
        "failed_files": failed_files,
        "skipped_unchanged": skipped_unchanged,
    }
    logger.info(
        "文档加载完成: records=%s failed_files=%s skipped_unchanged=%s",
        len(records),
        failed_files,
        skipped_unchanged,
    )
    if failed_files:
        logger.warning(f"失败文件列表: {failed_files}")
    return records, stats
