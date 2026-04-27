import json
import sqlite3
from pathlib import Path
from typing import Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from RAG.config.logger_runtime import get_logger

logger = get_logger("rag")


class VectorStore:
    """
    基于 LangChain FAISS 的向量库，并附带 metadata.json 与 sqlite 分块记录。
    建库与增量追加均须传入与运行时相同的 Embedding 封装（见 DP/embedding_service）。
    """

    def __init__(self, index_dir: str):
        # 索引根目录：其下 faiss_store/ 为 LangChain 持久化目录
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir = self.index_dir / "faiss_store"
        self.meta_path = self.index_dir / "metadata.json"
        self.sqlite_path = self.index_dir / "chunks.db"
        self._init_sqlite()

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

    def build_or_append(
        self, chunks: List[Document], embeddings_model, incremental: bool = True
    ) -> Dict:
        """
        将切分后的 Document 写入 FAISS；incremental 为真且已有索引时在原库上追加。
        embeddings_model：须含 .model（HuggingFaceEmbeddings 实例），与 RAG 检索端一致。
        """
        logger.info(
            "开始写入向量库: chunks=%s incremental=%s", len(chunks), incremental
        )
        all_meta: List[Dict] = []
        if incremental and self.faiss_dir.exists():
            # 加载已有向量库并追加文档（会重新编码新 chunk）
            vs = FAISS.load_local(
                str(self.faiss_dir),
                embeddings=embeddings_model.model,
                allow_dangerous_deserialization=True,
            )
            vs.add_documents(chunks)
            if self.meta_path.exists():
                all_meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            all_meta += [
                {"text": d.page_content, "metadata": d.metadata} for d in chunks
            ]
        else:
            # 全量新建索引
            vs = FAISS.from_documents(chunks, embedding=embeddings_model.model)
            all_meta = [
                {"text": d.page_content, "metadata": d.metadata} for d in chunks
            ]
        vs.save_local(str(self.faiss_dir))
        self.meta_path.write_text(
            json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._save_rows(chunks)
        logger.info("向量库写入完成: total=%s", len(all_meta))
        return {
            "total_chunks": len(all_meta),
            "faiss_dir": str(self.faiss_dir),
            "metadata_path": str(self.meta_path),
        }

    def _save_rows(self, chunks: List[Document]) -> None:
        """把本批 chunk 插入 sqlite，便于按来源追溯。"""
        with sqlite3.connect(self.sqlite_path) as conn:
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
