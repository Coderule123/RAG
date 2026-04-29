import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config_utils import load_config, setup_huggingface_env, setup_logger


CONFIG = load_config()
PATHS_CONFIG = CONFIG.get("paths", {})
VECTOR_CONFIG = CONFIG.get("vector_index", {})
MODELS_CONFIG = CONFIG.get("models", {})
HF_CONFIG = CONFIG.get("huggingface", {})
HF_RUNTIME = setup_huggingface_env(CONFIG)

# 重要：必须在导入 transformers/sentence-transformers 之前设置 HF 环境变量，
# 否则会继续请求默认的 huggingface.co 而非镜像端点。
from sentence_transformers import SentenceTransformer  # noqa: E402

LOADER_BY_SUFFIX = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
}


class DocumentProcessor:
    """文档前处理与向量索引构建器。"""

    def __init__(
        self,
        index_dir: str = PATHS_CONFIG.get("index_dir", "index_store"),
        embedding_model: str = MODELS_CONFIG.get("embedding_model", "BAAI/bge-small-zh-v1.5"),
        embedding_instance: Optional[SentenceTransformer] = None,
        index_type: str = VECTOR_CONFIG.get("index_type", "flat"),
        ivf_nlist: int = VECTOR_CONFIG.get("ivf_nlist", 1024),
        hnsw_m: int = VECTOR_CONFIG.get("hnsw_m", 32),
    ) -> None:
        """初始化路径、模型与索引参数。"""
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "docs.index"
        self.meta_path = self.index_dir / "metadata.json"
        self.registry_path = self.index_dir / "doc_registry.json"
        self.sqlite_path = self.index_dir / "chunks.db"
        self.embedding_model = embedding_instance or SentenceTransformer(
            embedding_model,
            local_files_only=bool(HF_CONFIG.get("local_files_only", False)),
            # transformers 新版本要求通过 model_kwargs 传 cache_dir（避免弃用警告）
            model_kwargs={"cache_dir": HF_RUNTIME["sentence_cache_dir"]},
        )
        self.index_type = index_type.lower()
        self.ivf_nlist = ivf_nlist
        self.hnsw_m = hnsw_m
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """初始化 SQLite 表，用于存储 chunk 元数据。"""
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    page TEXT,
                    chunk_id INTEGER,
                    doc_hash TEXT,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    @staticmethod
    def _sha256(path: Path) -> str:
        """计算文件 SHA256，用于增量更新判定。"""
        sha = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(block)
        return sha.hexdigest()

    def _load_registry(self) -> Dict[str, str]:
        """读取文件哈希注册表。"""
        if not self.registry_path.exists():
            return {}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _save_registry(self, registry: Dict[str, str]) -> None:
        """保存文件哈希注册表。"""
        self.registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_loader(self, file_path: Path):
        """根据后缀选择对应文档 Loader。"""
        suffix = file_path.suffix.lower()
        loader_cls = LOADER_BY_SUFFIX.get(suffix)
        if not loader_cls:
            return None
        if loader_cls is TextLoader:
            return loader_cls(str(file_path), encoding="utf-8", autodetect_encoding=True)
        return loader_cls(str(file_path))

    def load_documents(
        self, data_dir: str, incremental: bool = True
    ) -> Tuple[List[Dict], Dict[str, str], Dict[str, int]]:
        """加载文档并保留元数据，支持增量与单文件异常隔离。"""
        root = Path(data_dir)
        if not root.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        registry = self._load_registry()
        next_registry = dict(registry)
        records: List[Dict] = []
        skipped_unchanged = 0
        failed_files = 0
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            loader = self._get_loader(file_path)
            if loader is None:
                continue
            source = str(file_path.resolve())
            file_hash = self._sha256(file_path)
            if incremental and registry.get(source) == file_hash:
                skipped_unchanged += 1
                continue
            try:
                # 单文件失败不影响整体流程，直接记录并跳过。
                docs = loader.load()
            except Exception as exc:
                failed_files += 1
                logger.exception(f"加载文档失败，已跳过: {source}, error={exc}")
                continue
            for page_idx, doc in enumerate(docs):
                text = (doc.page_content or "").strip()
                if not text:
                    continue
                metadata = dict(doc.metadata)
                metadata["source"] = metadata.get("source", source)
                metadata["doc_hash"] = file_hash
                metadata["page"] = metadata.get("page", metadata.get("page_number", page_idx))
                records.append(
                    {
                        "text": text,
                        "metadata": metadata,
                    }
                )
            next_registry[source] = file_hash
        stats = {
            "skipped_unchanged": skipped_unchanged,
            "failed_files": failed_files,
        }
        return records, next_registry, stats

    @staticmethod
    def build_splitter(chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
        """构造按语义边界递进切分的 splitter。"""
        if chunk_size <= overlap:
            raise ValueError("chunk_size 必须大于 overlap")
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[
                "\n\n",
                "\n",
                "。",  # 中文句号，优先按语义边界切分
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

    def chunk_documents(
        self, docs: List[Dict], chunk_size: int = 500, overlap: int = 80
    ) -> List[Dict]:
        """将文档拆分为 chunk，并补齐 doc_id/chunk_id 元数据。"""
        splitter = self.build_splitter(chunk_size, overlap)
        chunks: List[Dict] = []
        for doc_idx, doc in enumerate(docs):
            parts = splitter.split_text(doc["text"])
            for chunk_idx, part in enumerate(parts):
                chunks.append(
                    {
                        "text": part,
                        "metadata": {
                            **doc["metadata"],
                            "doc_id": doc_idx,
                            "chunk_id": chunk_idx,
                        },
                    }
                )
        return chunks

    def _create_index(self, dim: int):
        """按配置创建 FAISS 索引实例。"""
        if self.index_type == "flat":
            return faiss.IndexFlatIP(dim)
        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            return index
        if self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            return faiss.IndexIVFFlat(quantizer, dim, self.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
        raise ValueError("index_type 仅支持: flat, ivf, hnsw")

    def _save_chunk_rows(self, chunks: List[Dict]) -> None:
        """将新增 chunk 元数据追加写入 SQLite。"""
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.executemany(
                """
                INSERT INTO chunks (source, page, chunk_id, doc_hash, text, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c["metadata"].get("source", ""),
                        str(c["metadata"].get("page", "")),
                        int(c["metadata"].get("chunk_id", -1)),
                        c["metadata"].get("doc_hash", ""),
                        c["text"],
                        json.dumps(c["metadata"], ensure_ascii=False),
                    )
                    for c in chunks
                ],
            )
            conn.commit()

    def build_index(
        self,
        chunks: List[Dict],
        batch_size: int = 128,
        incremental: bool = True,
    ) -> Dict:
        """分批编码并写入/追加 FAISS 索引。"""
        if not chunks:
            raise ValueError("没有可用于构建索引的 chunk")

        if len(chunks) > 500_000 and self.index_type == "flat":
            logger.warning("chunk 超过 50 万，建议切换 index_type=ivf/hnsw 提升检索速度")

        existing_meta: List[Dict] = []
        if incremental and self.index_path.exists() and self.meta_path.exists():
            index = faiss.read_index(str(self.index_path))
            existing_meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        else:
            index = None

        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [item["text"] for item in batch]
            # 分批向量化，降低大规模场景峰值内存。
            vectors = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings = np.array(vectors, dtype=np.float32)
            if index is None:
                index = self._create_index(embeddings.shape[1])
            if hasattr(index, "is_trained") and not index.is_trained:
                # IVF 等索引需要先训练再 add。
                index.train(embeddings)
            index.add(embeddings)
            added += len(batch)

        if index is None:
            raise ValueError("索引构建失败")
        faiss.write_index(index, str(self.index_path))
        all_meta = existing_meta + chunks if incremental else chunks
        self.meta_path.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self._save_chunk_rows(chunks)
        return {"added_chunks": added, "total_chunks": len(all_meta), "index_type": self.index_type}

    def ingest(
        self,
        data_dir: str,
        chunk_size: int = 500,
        overlap: int = 80,
        batch_size: int = 128,
        incremental: bool = True,
    ) -> Dict:
        """执行端到端入库：加载 -> 切分 -> 向量化 -> 索引与元数据落盘。"""
        docs, next_registry, stats = self.load_documents(data_dir, incremental=incremental)
        if not docs:
            self._save_registry(next_registry)
            return {
                "document_count": 0,
                "chunk_count": 0,
                "message": "没有新增或变更文档",
                **stats,
            }
        chunks = self.chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
        build_result = self.build_index(chunks, batch_size=batch_size, incremental=incremental)
        self._save_registry(next_registry)
        return {
            "document_count": len(docs),
            "chunk_count": build_result["added_chunks"],
            "total_chunks": build_result["total_chunks"],
            "index_type": build_result["index_type"],
            "index_path": str(self.index_path),
            "metadata_path": str(self.meta_path),
            "sqlite_path": str(self.sqlite_path),
            "registry_path": str(self.registry_path),
            **stats,
        }


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="文档预处理与向量索引构建")
    parser.add_argument("--data-dir", default=PATHS_CONFIG.get("data_dir", "data"), help="知识库目录")
    parser.add_argument("--index-dir", default=PATHS_CONFIG.get("index_dir", "index_store"), help="索引目录")
    parser.add_argument("--chunk-size", type=int, default=VECTOR_CONFIG.get("chunk_size", 500))
    parser.add_argument("--overlap", type=int, default=VECTOR_CONFIG.get("overlap", 80))
    parser.add_argument("--batch-size", type=int, default=VECTOR_CONFIG.get("batch_size", 128))
    parser.set_defaults(incremental=VECTOR_CONFIG.get("incremental", True))
    parser.add_argument("--incremental", dest="incremental", action="store_true", help="启用增量更新")
    parser.add_argument("--no-incremental", dest="incremental", action="store_false", help="禁用增量更新")
    parser.add_argument(
        "--index-type",
        default=VECTOR_CONFIG.get("index_type", "flat"),
        choices=["flat", "ivf", "hnsw"],
    )
    parser.add_argument(
        "--embedding-model",
        default=MODELS_CONFIG.get("embedding_model", "BAAI/bge-small-zh-v1.5"),
    )
    return parser


def main() -> None:
    """入口：默认纯配置运行；传参时按 CLI 覆盖。"""
    log_file = setup_logger(PATHS_CONFIG.get("doc_logs_dir", "./logs/doc"))
    logger.info(f"document_processor 启动，日志文件: {log_file}")
    logger.info(f"HF 运行配置: {json.dumps(HF_RUNTIME, ensure_ascii=False)}")

    if len(sys.argv) == 1:
        # 纯配置模式：无参数启动时完全按 config.yaml 执行。
        data_dir = PATHS_CONFIG.get("data_dir", "data")
        index_dir = PATHS_CONFIG.get("index_dir", "index_store")
        embedding_model = MODELS_CONFIG.get("embedding_model", "BAAI/bge-small-zh-v1.5")
        index_type = VECTOR_CONFIG.get("index_type", "flat")
        chunk_size = int(VECTOR_CONFIG.get("chunk_size", 500))
        overlap = int(VECTOR_CONFIG.get("overlap", 80))
        batch_size = int(VECTOR_CONFIG.get("batch_size", 128))
        incremental = bool(VECTOR_CONFIG.get("incremental", True))
        config_snapshot = {
            "data_dir": data_dir,
            "index_dir": index_dir,
            "embedding_model": embedding_model,
            "index_type": index_type,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "batch_size": batch_size,
            "incremental": incremental,
        }
        logger.info(f"生效配置摘要: {json.dumps(config_snapshot, ensure_ascii=False)}")
        processor = DocumentProcessor(
            index_dir=index_dir,
            embedding_model=embedding_model,
            index_type=index_type,
        )
        result = processor.ingest(
            data_dir=data_dir,
            chunk_size=chunk_size,
            overlap=overlap,
            batch_size=batch_size,
            incremental=incremental,
        )
    else:
        args = build_parser().parse_args()
        logger.info(
            f"CLI 覆盖参数: data_dir={args.data_dir}, index_dir={args.index_dir}, "
            f"chunk_size={args.chunk_size}, overlap={args.overlap}, batch_size={args.batch_size}, "
            f"incremental={args.incremental}, index_type={args.index_type}"
        )
        processor = DocumentProcessor(
            index_dir=args.index_dir,
            embedding_model=args.embedding_model,
            index_type=args.index_type,
        )
        result = processor.ingest(
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            incremental=args.incremental,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
