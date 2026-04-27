from pathlib import Path
from typing import List, Tuple

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
}


def get_loader(file_path: Path):
    """根据后缀构造对应 Loader；不支持的类型返回 None。"""
    suffix = file_path.suffix.lower()
    loader_cls = LOADER_BY_SUFFIX.get(suffix)
    if not loader_cls:
        return None
    if loader_cls is TextLoader:
        return loader_cls(str(file_path), encoding="utf-8", autodetect_encoding=True)
    return loader_cls(str(file_path))


def load_documents(data_dir: str) -> Tuple[List[Document], int]:
    """
    递归扫描 data_dir 下支持的文件，加载为 LangChain Document 列表。
    返回 (文档列表, 加载失败文件数)；每条记录补充 source / page 元数据便于追溯。
    """
    logger.info("开始加载文档: %s", data_dir)
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    records: List[Document] = []
    failed_files = 0
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        loader = get_loader(file_path)
        if not loader:
            continue
        try:
            docs = loader.load()
            logger.info(f"成功加载文档: {file_path}")
        except Exception as exc:
            failed_files += 1
            logger.exception("加载失败，跳过文件: %s, err=%s", str(file_path), exc)
            continue
        for page_idx, doc in enumerate(docs):
            text = (doc.page_content or "").strip()
            if not text:
                continue
            metadata = dict(doc.metadata)
            metadata["source"] = metadata.get("source", str(file_path.resolve()))
            metadata["page"] = metadata.get(
                "page", metadata.get("page_number", page_idx)
            )
            records.append(Document(page_content=text, metadata=metadata))
    logger.info("文档加载完成: records=%s failed_files=%s", len(records), failed_files)
    if failed_files:
        logger.warning(f"失败文件列表: {failed_files}")
    return records, failed_files
