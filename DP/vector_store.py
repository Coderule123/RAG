import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from RAG.config.logger_runtime import get_logger
from RAG.DP.document_loader import alternate_source_keys, normalize_source_key

logger = get_logger("rag")


def normalize_for_dedup(text: str) -> str:
    """统一空白与大小写，用于识别完全重复的内容。"""
    normalized = re.sub(r"\s+", "", text or "").lower()
    return normalized.strip()


class VectorStore:
    """
    基于 LangChain FAISS 的向量库，并附带 metadata/ 分文件元数据、doc_hash.json 与 sqlite 分块记录。
    建库与增量追加均须传入与运行时相同的 Embedding 封装（见 DP/embedding_service）。
    """

    # 与 LangChain FAISS.save_local / load_local 默认 index_name 一致
    _FAISS_INDEX_NAME = "index"

    def __init__(self, index_dir: str):
        # 索引根目录：其下 faiss_store/ 为 LangChain 持久化目录
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir = self.index_dir / "faiss_store"
        self.doc_hash_path = self.index_dir / "doc_hash.json"
        self.metadata_docs_dir = self.index_dir / "metadata"
        self.sqlite_path = self.index_dir / "chunks.db"
        self._init_sqlite()

    def _faiss_persisted_ready(self) -> bool:
        """是否存在可加载的 FAISS 持久化文件（目录存在但文件缺失时不视为可增量）。"""
        faiss_file = self.faiss_dir / f"{self._FAISS_INDEX_NAME}.faiss"
        pkl_file = self.faiss_dir / f"{self._FAISS_INDEX_NAME}.pkl"
        return faiss_file.is_file() and pkl_file.is_file()

    def _init_sqlite(self) -> None:
        """创建分块明细表，便于审计与排查（不参与向量检索）。"""
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    page TEXT,
                    chunk_id INTEGER,
                    text TEXT,
                    metadata_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def _load_store(self, embeddings_model):
        return FAISS.load_local(
            str(self.faiss_dir),
            embeddings=embeddings_model.model,
            allow_dangerous_deserialization=True,
        )

    @staticmethod
    def _iter_docstore(store) -> Iterable[Tuple[str, Document]]:
        doc_dict = getattr(store.docstore, "_dict", {})
        for doc_id, doc in doc_dict.items():
            if isinstance(doc, Document):
                yield doc_id, doc

    def _prune_existing(self, store, incoming_sources: set) -> Dict[str, int]:
        """
        删除增量更新中将被替换的旧来源 chunk，同时顺手清理历史重复 chunk。
        这样变化文件重新入库时不会在检索结果中保留旧内容。
        """
        ids_to_delete: List[str] = []
        seen_texts = set()
        removed_replaced = 0
        removed_duplicates = 0

        incoming_norm = {normalize_source_key(s) for s in incoming_sources if s}

        for doc_id, doc in self._iter_docstore(store):
            doc_src = doc.metadata.get("source", "")
            if doc_src and normalize_source_key(doc_src) in incoming_norm:
                ids_to_delete.append(doc_id)
                removed_replaced += 1
                continue

            text_key = normalize_for_dedup(doc.page_content)
            if not text_key:
                ids_to_delete.append(doc_id)
                removed_duplicates += 1
                continue

            if text_key in seen_texts:
                ids_to_delete.append(doc_id)
                removed_duplicates += 1
                continue
            seen_texts.add(text_key)

        if ids_to_delete:
            if not hasattr(store, "delete"):
                raise RuntimeError("当前 LangChain FAISS 版本不支持 delete，无法执行增量替换")
            store.delete(ids=ids_to_delete)

        return {
            "removed_replaced": removed_replaced,
            "removed_duplicates": removed_duplicates,
        }

    @staticmethod
    def _store_to_metadata(store) -> List[Dict]:
        records: List[Dict] = []
        for _, doc in VectorStore._iter_docstore(store):
            records.append({"text": doc.page_content, "metadata": dict(doc.metadata)})
        return records

    @staticmethod
    def _metadata_basename_for_source(source: str, basename_groups: Dict[str, List[str]]) -> str:
        """
        以加载文档的文件名（路径 basename）作为 metadata/ 下文件名；同名不同路径时加短哈希区分。
        """
        name = Path(source).name.strip() or "unknown"
        peers = basename_groups.get(name, [])
        if len(peers) <= 1:
            return name
        stem, suf = Path(name).stem, Path(name).suffix
        short = hashlib.sha256(source.encode("utf-8")).hexdigest()[:8]
        return f"{stem}__{short}{suf}" if stem else f"{short}{suf}"

    def _persist_doc_hash_and_split_metadata(self, all_meta: List[Dict]) -> None:
        """
        写入 doc_hash.json（全量 source -> doc_hash），并在 metadata/ 下按文档拆分 JSON，
        单文件内容为分块条目列表（仅 text + metadata），metadata 中不含 source、doc_hash；文件名为文档 basename。
        """
        doc_hashes: Dict[str, str] = {}
        by_source: Dict[str, List[Dict]] = defaultdict(list)

        for rec in all_meta:
            if not isinstance(rec, dict):
                continue
            meta = dict(rec.get("metadata") or {})
            source = meta.get("source", "") or ""
            if source:
                source = normalize_source_key(source)
            doc_hash = meta.get("doc_hash")
            if source and doc_hash:
                doc_hashes[source] = doc_hash
            meta_no_export = {
                k: v for k, v in meta.items() if k not in ("source", "doc_hash")
            }
            row = {"text": rec.get("text", ""), "metadata": meta_no_export}
            if source:
                by_source[source].append(row)

        self.doc_hash_path.write_text(
            json.dumps(doc_hashes, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self.metadata_docs_dir.mkdir(parents=True, exist_ok=True)
        for path in self.metadata_docs_dir.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)

        basename_groups: Dict[str, List[str]] = defaultdict(list)
        for src in by_source:
            basename_groups[Path(src).name].append(src)

        for source, rows in by_source.items():
            fname = self._metadata_basename_for_source(source, basename_groups)
            out_path = self.metadata_docs_dir / fname
            out_path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def _build_store_in_batches(
        self, chunks: List[Document], embeddings_model, batch_size: int
    ):
        if not chunks:
            return None

        first_batch = chunks[:batch_size]
        store = FAISS.from_documents(first_batch, embedding=embeddings_model.model)
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            store.add_documents(batch)
        return store

    def _replace_sqlite_rows(
        self, sources: set, chunks: List[Document], incremental: bool
    ) -> None:
        """移除被替换来源的旧明细，再写入本次新增 chunk。"""
        expanded: set[str] = set()
        for s in sources:
            expanded.update(alternate_source_keys(s))
        with sqlite3.connect(self.sqlite_path) as conn:
            if not incremental:
                conn.execute("DELETE FROM chunks")
            elif expanded:
                placeholders = ",".join("?" for _ in expanded)
                conn.execute(
                    f"DELETE FROM chunks WHERE source IN ({placeholders})",
                    tuple(expanded),
                )
            if chunks:
                conn.executemany(
                    """
                    INSERT INTO chunks (source, page, chunk_id, text, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            c.metadata.get("source", ""),
                            str(c.metadata.get("page", "")),
                            int(c.metadata.get("chunk_id", -1)),
                            c.page_content,
                            json.dumps(c.metadata, ensure_ascii=False),
                        )
                        for c in chunks
                    ],
                )
            conn.commit()

    def build_or_append(
        self,
        chunks: List[Document],
        embeddings_model,
        incremental: bool = True,
        batch_size: int = 128,
    ) -> Dict:
        """
        将切分后的 Document 写入 FAISS；incremental 为真且已有索引时在原库上追加。
        embeddings_model：须含 .model（HuggingFaceEmbeddings 实例），与 RAG 检索端一致。
        """
        logger.info(
            "开始写入向量库: chunks=%s incremental=%s batch_size=%s",
            len(chunks),
            incremental,
            batch_size,
        )
        if not chunks:
            raise ValueError("没有可写入向量库的 chunk")

        incoming_sources = {
            normalize_source_key(s)
            for chunk in chunks
            if (s := chunk.metadata.get("source", ""))
        }
        removed_stats = {
            "removed_replaced": 0,
            "removed_duplicates": 0,
        }

        use_incremental_append = incremental and self._faiss_persisted_ready()
        if incremental and not self._faiss_persisted_ready():
            logger.info(
                "增量模式已开启但未找到完整 FAISS 文件 (%s.faiss / .pkl)，按首次建库处理: %s",
                self._FAISS_INDEX_NAME,
                self.faiss_dir,
            )

        if use_incremental_append:
            vs = self._load_store(embeddings_model)
            removed_stats = self._prune_existing(vs, incoming_sources)
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                vs.add_documents(batch)
        else:
            vs = self._build_store_in_batches(chunks, embeddings_model, batch_size)

        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(self.faiss_dir))
        all_meta = self._store_to_metadata(vs)
        self._persist_doc_hash_and_split_metadata(all_meta)
        self._replace_sqlite_rows(incoming_sources, chunks, use_incremental_append)
        logger.info(
            "向量库写入完成: total=%s removed_replaced=%s removed_duplicates=%s",
            len(all_meta),
            removed_stats["removed_replaced"],
            removed_stats["removed_duplicates"],
        )
        return {
            "added_chunks": len(chunks),
            "total_chunks": len(all_meta),
            "faiss_dir": str(self.faiss_dir),
            "doc_hash_path": str(self.doc_hash_path),
            "metadata_docs_dir": str(self.metadata_docs_dir),
            **removed_stats,
        }
