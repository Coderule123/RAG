"""
RAG 知识库管理 Web 服务：文档管理、向量查看/删除、一键建库。
手动启动，不随 run.sh 启动：

    cd chat_assistant
    python3 -m RAG.web.rag_web_server --host 0.0.0.0 --port 17892
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import shutil
import signal
import sqlite3
import sys
import threading
import time
import traceback
import urllib.parse
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from RAG.config.config_runtime import load_config
from RAG.config.hf_runtime import setup_huggingface_env
from RAG.config.logger_runtime import get_logger, setup_logging
from RAG.DP.document_loader import (
    SUPPORTED_SUFFIXES,
    canonical_source_path,
    compute_file_hash,
    load_documents,
    load_docx_documents,
    load_hash_registry,
    load_pdf_documents,
    normalize_source_key,
    resolve_doc_tag,
)
from RAG.DP.embedding_service import EmbeddingService
from RAG.DP.semantic_splitter import split_documents
from RAG.DP.vector_store import VectorStore

MAX_PREVIEW_BYTES = 512 * 1024
MAX_PREVIEW_CHARS = 120_000


def get_default_html_path() -> Path:
    return Path(__file__).resolve().with_name("rag_web_server.html")


def get_default_js_path() -> Path:
    return Path(__file__).resolve().with_name("rag_web_server.js")


def iso_mtime(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def load_index_html(html_path: Path) -> str:
    try:
        return html_path.read_text(encoding="utf-8")
    except OSError as e:
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>RAG Admin</title></head><body>"
            f"<h3>rag_web_server.html not found</h3><p>{html_path}</p><pre>{e}</pre>"
            "</body></html>"
        )


def safe_relpath(relpath: str) -> str:
    rel = (relpath or "").strip().replace("\\", "/").lstrip("/")
    if not rel or ".." in rel.split("/"):
        raise ValueError("invalid relative path")
    return rel


def safe_path_under(root: Path, relpath: str) -> Path:
    rel = safe_relpath(relpath)
    file_path = (root / rel).resolve()
    root_resolved = root.resolve()
    if file_path != root_resolved and root_resolved not in file_path.parents:
        raise ValueError("path escapes root")
    return file_path


def resolve_config_paths(config: dict) -> dict[str, Path]:
    paths = config.get("paths", {})
    return {
        "data_dir": Path(paths.get("data_dir", "./RAG/assets/data")).expanduser().resolve(),
        "index_dir": Path(paths.get("index_dir", "./RAG/assets/index_store")).expanduser().resolve(),
        "doc_logs_dir": Path(paths.get("doc_logs_dir", "./RAG/logs/doc")).expanduser().resolve(),
        "visitor_state_dir": Path(
            paths.get("visitor_state_dir", "./RAG/assets/visitor_states")
        )
        .expanduser()
        .resolve(),
    }


def iso_ts(ts: Any) -> Optional[str]:
    if not isinstance(ts, (int, float)):
        return None
    return datetime.fromtimestamp(float(ts)).isoformat(timespec="seconds")


def safe_visitor_path(states_dir: Path, face_id: str) -> Path:
    if not face_id or "/" in face_id or "\\" in face_id or ".." in face_id:
        raise ValueError("invalid face id")
    file_path = (states_dir / f"{face_id}.json").resolve()
    states_root = states_dir.resolve()
    if file_path.parent != states_root:
        raise ValueError("invalid visitor path")
    return file_path


def _build_step_view(steps: list) -> tuple[list[dict[str, Any]], Optional[dict], int, bool]:
    """Normalize tour steps and compute current / progress."""
    normalized: list[dict[str, Any]] = []
    for raw in steps:
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        normalized.append(
            {
                "id": str(raw["id"]),
                "title": str(raw.get("title") or raw["id"]),
                "order": int(raw.get("order") or 0),
                "asked": bool(raw.get("asked", False)),
                "asked_at": iso_ts(raw.get("asked_at")),
            }
        )
    normalized.sort(key=lambda s: s["order"])

    current_step: Optional[dict] = None
    for step in normalized:
        if not step["asked"]:
            current_step = {
                "id": step["id"],
                "title": step["title"],
                "order": step["order"],
            }
            break

    for step in normalized:
        if step["asked"]:
            step["status"] = "done"
        elif current_step and step["id"] == current_step["id"]:
            step["status"] = "current"
        else:
            step["status"] = "pending"

    asked_count = sum(1 for step in normalized if step["asked"])
    all_done = bool(normalized) and current_step is None
    return normalized, current_step, asked_count, all_done


def _build_interest_profile_view(raw_profile: Any) -> dict[str, Any]:
    """Normalize interest_profile for web display."""
    profile = raw_profile if isinstance(raw_profile, dict) else {}
    last_tag = profile.get("last_vehicle_tag")
    last_vehicle_tag = (
        str(last_tag).strip().lower()
        if isinstance(last_tag, str) and last_tag.strip()
        else None
    )

    vehicles_out: list[dict[str, Any]] = []
    vehicles_raw = profile.get("vehicles") if isinstance(profile.get("vehicles"), dict) else {}
    for tag, info in vehicles_raw.items():
        if not isinstance(tag, str) or not tag.strip() or not isinstance(info, dict):
            continue
        topics_raw = info.get("topics") if isinstance(info.get("topics"), dict) else {}
        topics: list[dict[str, Any]] = []
        for topic_id, topic_info in topics_raw.items():
            if not isinstance(topic_id, str) or not topic_id.strip():
                continue
            if not isinstance(topic_info, dict):
                continue
            try:
                count = max(1, int(topic_info.get("count", 1)))
            except (TypeError, ValueError):
                count = 1
            topics.append(
                {
                    "id": topic_id.strip(),
                    "title": str(topic_info.get("title") or topic_id),
                    "count": count,
                    "first_asked_at": iso_ts(topic_info.get("first_asked_at")),
                    "last_asked_at": iso_ts(topic_info.get("last_asked_at")),
                }
            )
        topics.sort(key=lambda t: (-t["count"], t["title"]))
        try:
            ask_count = max(1, int(info.get("ask_count", 1)))
        except (TypeError, ValueError):
            ask_count = 1
        vehicles_out.append(
            {
                "tag": tag.strip().lower(),
                "ask_count": ask_count,
                "first_asked_at": iso_ts(info.get("first_asked_at")),
                "last_asked_at": iso_ts(info.get("last_asked_at")),
                "topic_count": len(topics),
                "topics": topics,
            }
        )

    vehicles_out.sort(
        key=lambda v: (v.get("last_asked_at") or "", v.get("ask_count") or 0),
        reverse=True,
    )
    return {
        "last_vehicle_tag": last_vehicle_tag,
        "vehicle_count": len(vehicles_out),
        "topic_count": sum(v["topic_count"] for v in vehicles_out),
        "vehicles": vehicles_out,
    }


def summarize_visitor_state(raw: dict, face_id: str, mtime: float) -> dict[str, Any]:
    tour = raw.get("tour_process") if isinstance(raw.get("tour_process"), dict) else {}
    steps, current_step, asked_count, all_done = _build_step_view(
        tour.get("steps") if isinstance(tour.get("steps"), list) else []
    )
    interest = _build_interest_profile_view(raw.get("interest_profile"))
    ask_name = raw.get("ask_name") if isinstance(raw.get("ask_name"), dict) else {}
    person_name = raw.get("person_name")
    return {
        "face_id": str(raw.get("vision_user_id") or face_id),
        "person_name": str(person_name) if person_name else None,
        "ask_name_asked": bool(ask_name.get("asked", False)),
        "ask_name_first_asked_at": iso_ts(ask_name.get("first_asked_at")),
        "asked_count": asked_count,
        "total_steps": len(steps),
        "current_step": current_step,
        "all_done": all_done,
        "last_vehicle_tag": interest["last_vehicle_tag"],
        "vehicle_count": interest["vehicle_count"],
        "topic_count": interest["topic_count"],
        "created_at": iso_ts(raw.get("created_at")),
        "updated_at": iso_ts(raw.get("updated_at")) or iso_mtime(mtime),
        "mtime": iso_mtime(mtime),
    }


def load_visitor_detail(raw: dict, face_id: str, mtime: float) -> dict[str, Any]:
    summary = summarize_visitor_state(raw, face_id, mtime)
    tour = raw.get("tour_process") if isinstance(raw.get("tour_process"), dict) else {}
    steps, _, _, _ = _build_step_view(
        tour.get("steps") if isinstance(tour.get("steps"), list) else []
    )
    summary["steps"] = steps
    summary["interest_profile"] = _build_interest_profile_view(raw.get("interest_profile"))
    return summary


def list_visitors(states_dir: Path) -> list[dict[str, Any]]:
    visitors: list[dict[str, Any]] = []
    if not states_dir.exists():
        return visitors
    for item in states_dir.iterdir():
        if not item.is_file() or item.suffix != ".json":
            continue
        face_id = item.stem
        try:
            raw = json.loads(item.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        visitors.append(summarize_visitor_state(raw, face_id, item.stat().st_mtime))
    visitors.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    return visitors


def list_data_files(data_dir: Path, index_dir: Path) -> list[dict[str, Any]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    doc_hashes = load_hash_registry(str(index_dir))
    files: list[dict[str, Any]] = []

    for file_path in sorted(data_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        rel = file_path.relative_to(data_dir).as_posix()
        stat = file_path.stat()
        source = canonical_source_path(file_path)
        source_norm = normalize_source_key(source)
        local_hash = compute_file_hash(file_path)
        indexed_hash = doc_hashes.get(source_norm)
        if indexed_hash is None:
            index_status = "not_indexed"
        elif indexed_hash == local_hash:
            index_status = "indexed"
        else:
            index_status = "changed"
        tag = resolve_doc_tag(file_path, data_dir)
        files.append(
            {
                "relpath": rel,
                "tag": tag,
                "source": source_norm,
                "size": stat.st_size,
                "mtime": iso_mtime(stat.st_mtime),
                "index_status": index_status,
            }
        )
    return files


def list_tags(data_dir: Path) -> list[str]:
    tags: set[str] = {"general"}
    if not data_dir.is_dir():
        return sorted(tags)
    for child in data_dir.iterdir():
        if child.is_dir():
            tags.add(child.name)
    return sorted(tags)


def read_doc_hash(index_dir: Path) -> dict[str, str]:
    path = index_dir / "doc_hash.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {normalize_source_key(str(k)): str(v) for k, v in payload.items() if k and v}
    except Exception:
        pass
    return {}


def faiss_ready(index_dir: Path) -> bool:
    faiss_dir = index_dir / "faiss_store"
    return (faiss_dir / "index.faiss").is_file() and (faiss_dir / "index.pkl").is_file()


def count_sqlite_chunks(index_dir: Path) -> int:
    db_path = index_dir / "chunks.db"
    if not db_path.is_file():
        return 0
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return int(row[0]) if row else 0
    except Exception:
        return 0


def list_metadata_files(index_dir: Path) -> list[str]:
    meta_dir = index_dir / "metadata"
    if not meta_dir.is_dir():
        return []
    out: list[str] = []
    for p in sorted(meta_dir.rglob("*.json")):
        out.append(p.relative_to(meta_dir).as_posix())
    return out


def metadata_doc_name(metadata_path: str) -> str:
    """metadata 路径对应的展示名（去掉 .json 后缀）。"""
    name = Path(metadata_path).name
    if name.lower().endswith(".json"):
        return name[:-5]
    return name


def group_metadata_by_document(index_dir: Path) -> list[dict[str, Any]]:
    """按 metadata 文件（镜像 data 文档）分组，返回完整 text。"""
    meta_dir = index_dir / "metadata"
    if not meta_dir.is_dir():
        return []
    documents: list[dict[str, Any]] = []
    for meta_file in sorted(meta_dir.rglob("*.json")):
        rel_meta = meta_file.relative_to(meta_dir).as_posix()
        try:
            rows = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        chunks: list[dict[str, Any]] = []
        tag = ""
        for row in rows:
            if not isinstance(row, dict):
                continue
            meta = row.get("metadata") or {}
            text = str(row.get("text", ""))
            row_tag = str(meta.get("tag", "") or "")
            if row_tag and not tag:
                tag = row_tag
            chunks.append(
                {
                    "doc_id": meta.get("doc_id"),
                    "chunk_id": meta.get("chunk_id"),
                    "tag": row_tag,
                    "page": meta.get("page", ""),
                    "text": text,
                }
            )
        documents.append(
            {
                "metadata_path": rel_meta,
                "doc_name": metadata_doc_name(rel_meta),
                "tag": tag,
                "chunk_count": len(chunks),
                "chunks": chunks,
            }
        )
    return documents


def read_data_file_preview(data_dir: Path, relpath: str) -> dict[str, Any]:
    """读取 data 目录下文档内容用于预览。"""
    file_path = safe_path_under(data_dir, relpath)
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {relpath}")

    suffix = file_path.suffix.lower()
    stat = file_path.stat()
    truncated = False

    if suffix in (".txt", ".md", ".csv"):
        raw = file_path.read_bytes()
        if len(raw) > MAX_PREVIEW_BYTES:
            raw = raw[:MAX_PREVIEW_BYTES]
            truncated = True
        content = raw.decode("utf-8", errors="replace")
        if len(content) > MAX_PREVIEW_CHARS:
            content = content[:MAX_PREVIEW_CHARS]
            truncated = True
        return {
            "relpath": relpath,
            "format": "text",
            "content": content,
            "truncated": truncated,
            "size": stat.st_size,
        }

    if suffix == ".json":
        raw = file_path.read_bytes()
        if len(raw) > MAX_PREVIEW_BYTES:
            raw = raw[:MAX_PREVIEW_BYTES]
            truncated = True
        text = raw.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
            content = json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            content = text
        if len(content) > MAX_PREVIEW_CHARS:
            content = content[:MAX_PREVIEW_CHARS]
            truncated = True
        return {
            "relpath": relpath,
            "format": "json",
            "content": content,
            "truncated": truncated,
            "size": stat.st_size,
        }

    if suffix in (".pdf", ".docx"):
        file_hash = compute_file_hash(file_path)
        if suffix == ".pdf":
            docs = load_pdf_documents(file_path, file_hash)
        else:
            docs = load_docx_documents(file_path, file_hash)
        parts: list[str] = []
        total_len = 0
        for i, doc in enumerate(docs):
            page = doc.metadata.get("page", doc.metadata.get("source", ""))
            header = f"--- 片段 {i + 1}"
            if page not in ("", None):
                header += f" (page={page})"
            header += " ---"
            body = doc.page_content or ""
            block = f"{header}\n{body}"
            if total_len + len(block) > MAX_PREVIEW_CHARS:
                remain = MAX_PREVIEW_CHARS - total_len
                if remain > 0:
                    parts.append(block[:remain])
                truncated = True
                break
            parts.append(block)
            total_len += len(block) + 2
        return {
            "relpath": relpath,
            "format": "extracted",
            "content": "\n\n".join(parts),
            "truncated": truncated,
            "size": stat.st_size,
        }

    raise ValueError(f"unsupported preview type: {suffix}")


def read_metadata_detail(index_dir: Path, meta_relpath: str) -> Any:
    rel = safe_relpath(meta_relpath)
    if not rel.endswith(".json"):
        rel = f"{rel}.json"
    meta_file = safe_path_under(index_dir / "metadata", rel)
    if not meta_file.is_file():
        raise FileNotFoundError(f"metadata not found: {rel}")
    return json.loads(meta_file.read_text(encoding="utf-8"))


def read_log_incremental(
    file_path: Path, start_offset: int, max_bytes: int = 256 * 1024
) -> tuple[str, int, bool]:
    if start_offset < 0:
        start_offset = 0
    with file_path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        if start_offset > file_size:
            start_offset = 0
        f.seek(start_offset)
        data = f.read(max_bytes)
        next_offset = start_offset + len(data)
        has_more = next_offset < file_size
    return data.decode("utf-8", errors="replace"), next_offset, has_more


def latest_doc_log(doc_logs_dir: Path) -> Optional[Path]:
    if not doc_logs_dir.is_dir():
        return None
    logs = [p for p in doc_logs_dir.iterdir() if p.is_file()]
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def parse_multipart_form(
    content_type: str, body: bytes
) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    match = re.search(r"boundary=([^;\s]+)", content_type or "", re.I)
    if not match:
        raise ValueError("missing multipart boundary")
    boundary = match.group(1).strip().strip('"').encode("utf-8")
    delimiter = b"--" + boundary
    parts = body.split(delimiter)
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}

    for part in parts:
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        header_blob, _, content = part.partition(b"\r\n\r\n")
        if not content:
            continue
        content = content.rstrip(b"\r\n")
        headers = header_blob.decode("utf-8", errors="replace")
        name_match = re.search(r'name="([^"]+)"', headers)
        if not name_match:
            continue
        name = name_match.group(1)
        filename_match = re.search(r'filename="([^"]*)"', headers)
        if filename_match is not None:
            filename = filename_match.group(1) or "upload.bin"
            files[name] = (filename, content)
        else:
            fields[name] = content.decode("utf-8", errors="replace")
    return fields, files


def sanitize_filename(name: str) -> str:
    base = Path(name).name
    base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base).strip()
    return base or "upload.bin"


def sanitize_tag(tag: str) -> str:
    tag = (tag or "general").strip().replace("\\", "/").strip("/")
    if not tag or ".." in tag.split("/"):
        return "general"
    return tag


class JobRunner:
    def __init__(self, config: dict, paths: dict[str, Path]):
        self.config = config
        self.paths = paths
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "status": "idle",
            "kind": None,
            "message": "",
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def is_busy(self) -> bool:
        with self._lock:
            return self._state.get("status") == "running"

    def submit(self, kind: str, payload: dict[str, Any]) -> tuple[bool, str]:
        with self._lock:
            if self._state.get("status") == "running":
                return False, "已有任务正在执行，请稍候"
            self._state = {
                "status": "running",
                "kind": kind,
                "message": "任务启动中…",
                "started_at": iso_mtime(time.time()),
                "finished_at": None,
                "result": None,
                "error": None,
            }
        thread = threading.Thread(
            target=self._run_job, args=(kind, payload), daemon=True
        )
        thread.start()
        return True, "任务已提交"

    def _set_running_message(self, message: str) -> None:
        with self._lock:
            if self._state.get("status") == "running":
                self._state["message"] = message

    def _finish_success(self, result: Any) -> None:
        with self._lock:
            self._state.update(
                {
                    "status": "success",
                    "message": "任务完成",
                    "finished_at": iso_mtime(time.time()),
                    "result": result,
                    "error": None,
                }
            )

    def _finish_error(self, error: str) -> None:
        with self._lock:
            self._state.update(
                {
                    "status": "error",
                    "message": "任务失败",
                    "finished_at": iso_mtime(time.time()),
                    "result": None,
                    "error": error,
                }
            )

    def _run_job(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            if kind == "build":
                self._run_build(payload)
            elif kind == "delete":
                self._run_delete(payload)
            else:
                raise ValueError(f"unknown job kind: {kind}")
        except Exception as exc:
            logger = get_logger("rag")
            logger.exception("RAG web job failed: %s", kind)
            self._finish_error(f"{exc}\n{traceback.format_exc()}")

    def _run_build(self, payload: dict[str, Any]) -> None:
        paths = self.config.get("paths", {})
        vector = self.config.get("vector_index", {})
        models = self.config.get("models", {})
        hf_cfg = self.config.get("huggingface", {})

        incremental = payload.get("incremental")
        if incremental is None:
            incremental = bool(vector.get("incremental", True))

        data_dir = str(self.paths["data_dir"])
        index_dir = str(self.paths["index_dir"])

        setup_logging(
            paths.get("doc_logs_dir", "./RAG/logs/doc"),
            logger_name="rag",
            log_mode="timestamp",
        )
        self._set_running_message("初始化 HuggingFace 环境…")
        hf_runtime = setup_huggingface_env(self.config)

        self._set_running_message("扫描文档…")
        docs, load_stats = load_documents(data_dir, incremental=incremental, index_dir=index_dir)
        if not docs:
            result = {
                "document_count": 0,
                "chunk_count": 0,
                "total_chunks": 0,
                "message": "没有新增或变更文档",
                "index_dir": index_dir,
                **load_stats,
            }
            self._finish_success(result)
            return

        self._set_running_message("加载 Embedding 模型…")
        embed = EmbeddingService(
            model_name=models.get("embedding_model", "BAAI/bge-small-zh-v1.5"),
            sentence_cache_dir=hf_runtime["sentence_cache_dir"],
            local_files_only=bool(hf_cfg.get("local_files_only", False)),
        )

        self._set_running_message(f"语义切分 {len(docs)} 篇文档…")
        chunks = split_documents(
            docs,
            int(payload.get("chunk_size", vector.get("chunk_size", 500))),
            int(payload.get("overlap", vector.get("overlap", 80))),
            max_chunk_size=vector.get("max_chunk_size"),
            min_chunk_chars=int(vector.get("min_chunk_chars", 15)),
            add_context_header=bool(vector.get("add_context_header", True)),
            embeddings_model=embed,
            semantic_split=bool(vector.get("semantic_split", True)),
            semantic_breakpoint_percentile=float(
                vector.get("semantic_breakpoint_percentile", 88)
            ),
        )

        self._set_running_message(f"写入向量库 {len(chunks)} 个 chunk…")
        store = VectorStore(index_dir=index_dir, data_dir=data_dir)
        result = store.build_or_append(
            chunks,
            embed,
            incremental=incremental,
            batch_size=int(payload.get("batch_size", vector.get("batch_size", 128))),
        )
        result.update(
            {
                "document_count": len(docs),
                "chunk_count": result.get("added_chunks", len(chunks)),
                **load_stats,
            }
        )
        self._finish_success(result)

    def _run_delete(self, payload: dict[str, Any]) -> None:
        paths = self.config.get("paths", {})
        models = self.config.get("models", {})
        hf_cfg = self.config.get("huggingface", {})
        index_dir = str(self.paths["index_dir"])

        setup_logging(
            paths.get("doc_logs_dir", "./RAG/logs/doc"),
            logger_name="rag",
            log_mode="timestamp",
        )
        self._set_running_message("加载 Embedding 模型…")
        hf_runtime = setup_huggingface_env(self.config)
        embed = EmbeddingService(
            model_name=models.get("embedding_model", "BAAI/bge-small-zh-v1.5"),
            sentence_cache_dir=hf_runtime["sentence_cache_dir"],
            local_files_only=bool(hf_cfg.get("local_files_only", False)),
        )

        self._set_running_message("删除向量并重写索引…")
        store = VectorStore(index_dir=index_dir, data_dir=str(self.paths["data_dir"]))
        result = store.delete_by_metadata_selector(
            embed,
            source=payload.get("source"),
            metadata_json=payload.get("metadata"),
            delete_all=bool(payload.get("all")),
            doc_ids=payload.get("doc_ids"),
            chunk_id=payload.get("chunk_id"),
        )
        self._finish_success(result)


class RagAdminHandler(BaseHTTPRequestHandler):
    server_version = "RAGAdmin/1.0"

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(
        self,
        payload: str,
        status: int = HTTPStatus.OK,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        body = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _config(self) -> dict:
        return self.server.rag_config  # type: ignore[attr-defined]

    def _paths(self) -> dict[str, Path]:
        return self.server.rag_paths  # type: ignore[attr-defined]

    def _jobs(self) -> JobRunner:
        return self.server.job_runner  # type: ignore[attr-defined]

    def _html_path(self) -> Path:
        return self.server.html_path  # type: ignore[attr-defined]

    def _js_path(self) -> Path:
        return self.server.js_path  # type: ignore[attr-defined]

    def _index_html(self) -> str:
        return self.server.index_html  # type: ignore[attr-defined]

    def _drain_body(self) -> None:
        raw_len = self.headers.get("Content-Length", "0")
        try:
            content_len = int(raw_len)
        except ValueError:
            content_len = 0
        if content_len > 0:
            self.rfile.read(content_len)

    def _read_body_bytes(self) -> bytes:
        raw_len = self.headers.get("Content-Length", "0")
        try:
            content_len = int(raw_len)
        except ValueError:
            raise ValueError("invalid content length")
        if content_len <= 0:
            return b""
        return self.rfile.read(content_len)

    def _read_json_body(self) -> dict:
        raw = self._read_body_bytes()
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            raise ValueError("invalid json body")
        if not isinstance(data, dict):
            raise ValueError("json body must be object")
        return data

    def _parse_path(self) -> tuple[str, list[str], dict]:
        parsed = urllib.parse.urlsplit(self.path)
        clean_path = posixpath.normpath(urllib.parse.unquote(parsed.path))
        parts = [x for x in clean_path.split("/") if x]
        query = urllib.parse.parse_qs(parsed.query)
        return clean_path, parts, query

    def do_GET(self) -> None:
        path, parts, query = self._parse_path()
        paths = self._paths()
        config = self._config()

        if path == "/":
            self._send_text(self._index_html(), content_type="text/html; charset=utf-8")
            return

        if path == "/rag_web_server.js":
            js_path = self._js_path()
            if not js_path.is_file():
                self._send_json(
                    {"error": f"js not found: {js_path}"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            self._send_text(
                js_path.read_text(encoding="utf-8"),
                content_type="application/javascript; charset=utf-8",
            )
            return

        if path == "/healthz":
            self._send_json(
                {
                    "ok": True,
                    "data_dir": str(paths["data_dir"]),
                    "index_dir": str(paths["index_dir"]),
                    "doc_logs_dir": str(paths["doc_logs_dir"]),
                    "visitor_state_dir": str(paths["visitor_state_dir"]),
                    "html_path": str(self._html_path()),
                    "js_path": str(self._js_path()),
                }
            )
            return

        if path == "/api/visitors":
            states_dir = paths["visitor_state_dir"]
            self._send_json(
                {
                    "visitors": list_visitors(states_dir),
                    "visitor_state_dir": str(states_dir),
                }
            )
            return

        if len(parts) == 3 and parts[0] == "api" and parts[1] == "visitors":
            face_id = parts[2]
            try:
                file_path = safe_visitor_path(paths["visitor_state_dir"], face_id)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return
            if not file_path.is_file():
                self._send_json(
                    {"error": f"visitor not found: {face_id}"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            try:
                raw = json.loads(file_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                self._send_json(
                    {"error": f"failed to read visitor state: {e}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            if not isinstance(raw, dict):
                self._send_json(
                    {"error": "invalid visitor state format"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self._send_json(
                load_visitor_detail(raw, face_id, file_path.stat().st_mtime)
            )
            return

        if path == "/api/data/files":
            files = list_data_files(paths["data_dir"], paths["index_dir"])
            self._send_json(
                {
                    "files": files,
                    "tags": list_tags(paths["data_dir"]),
                    "data_dir": str(paths["data_dir"]),
                }
            )
            return

        if len(parts) >= 4 and parts[0] == "api" and parts[1] == "data" and parts[2] == "preview":
            relpath = "/".join(parts[3:])
            try:
                preview = read_data_file_preview(paths["data_dir"], relpath)
            except FileNotFoundError as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.NOT_FOUND)
                return
            except ValueError as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(preview)
            return

        if path == "/api/index/summary":
            index_dir = paths["index_dir"]
            doc_hash = read_doc_hash(index_dir)
            self._send_json(
                {
                    "index_dir": str(index_dir),
                    "faiss_ready": faiss_ready(index_dir),
                    "doc_hash_count": len(doc_hash),
                    "metadata_files": list_metadata_files(index_dir),
                    "chunk_count": count_sqlite_chunks(index_dir),
                    "vector_index": config.get("vector_index", {}),
                    "models": config.get("models", {}),
                }
            )
            return

        if path == "/api/index/metadata":
            index_dir = paths["index_dir"]
            meta_path = (query.get("path") or [""])[0]
            if meta_path:
                try:
                    detail = read_metadata_detail(index_dir, meta_path)
                except FileNotFoundError as e:
                    self._send_json({"error": str(e)}, status=HTTPStatus.NOT_FOUND)
                    return
                except ValueError as e:
                    self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._send_json({"path": meta_path, "chunks": detail})
                return
            documents = group_metadata_by_document(index_dir)
            total_chunks = sum(d.get("chunk_count", 0) for d in documents)
            self._send_json(
                {"documents": documents, "total_documents": len(documents), "total_chunks": total_chunks}
            )
            return

        if path == "/api/jobs/current":
            self._send_json({"job": self._jobs().snapshot()})
            return

        if path == "/api/logs/doc":
            doc_logs_dir = paths["doc_logs_dir"]
            log_file = latest_doc_log(doc_logs_dir)
            if not log_file:
                self._send_json(
                    {
                        "name": None,
                        "content": "",
                        "next_offset": 0,
                        "has_more": False,
                    }
                )
                return
            try:
                start_offset = int((query.get("from") or ["0"])[0])
            except ValueError:
                self._send_json(
                    {"error": "invalid from offset"}, status=HTTPStatus.BAD_REQUEST
                )
                return
            content, next_offset, has_more = read_log_incremental(log_file, start_offset)
            self._send_json(
                {
                    "name": log_file.name,
                    "content": content,
                    "next_offset": next_offset,
                    "has_more": has_more,
                }
            )
            return

        self._send_json({"error": f"unknown route: {path}"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path, parts, _ = self._parse_path()
        paths = self._paths()

        if path == "/api/data/upload":
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._send_json(
                    {"error": "Content-Type must be multipart/form-data"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            try:
                body = self._read_body_bytes()
                fields, files = parse_multipart_form(content_type, body)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return

            upload = files.get("file")
            if not upload:
                self._send_json(
                    {"error": "missing file field"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            filename, content = upload
            filename = sanitize_filename(filename)
            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                self._send_json(
                    {"error": f"unsupported file type: {suffix}"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            tag = sanitize_tag(fields.get("tag", "general"))
            dest_dir = paths["data_dir"] / tag
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / filename
            dest_path.write_bytes(content)
            rel = dest_path.relative_to(paths["data_dir"]).as_posix()
            self._send_json(
                {
                    "ok": True,
                    "relpath": rel,
                    "tag": tag,
                    "source": canonical_source_path(dest_path),
                }
            )
            return

        if path == "/api/data/move":
            try:
                body = self._read_json_body()
            except ValueError as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return
            src_rel = body.get("from")
            dst_rel = body.get("to")
            if not isinstance(src_rel, str) or not isinstance(dst_rel, str):
                self._send_json(
                    {"error": "from and to must be strings"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            try:
                src_path = safe_path_under(paths["data_dir"], src_rel)
                dst_path = safe_path_under(paths["data_dir"], dst_rel)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return
            if not src_path.is_file():
                self._send_json(
                    {"error": f"source not found: {src_rel}"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
            new_rel = dst_path.relative_to(paths["data_dir"]).as_posix()
            self._send_json({"ok": True, "relpath": new_rel})
            return

        if path == "/api/index/build":
            try:
                body = self._read_json_body()
            except ValueError as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return
            ok, message = self._jobs().submit("build", body)
            if not ok:
                self._send_json({"error": message}, status=HTTPStatus.CONFLICT)
                return
            self._send_json({"ok": True, "message": message})
            return

        if path == "/api/index/delete":
            try:
                body = self._read_json_body()
            except ValueError as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return
            metadata = (body.get("metadata") or "").strip()
            source = (body.get("source") or "").strip()
            if bool(metadata) == bool(source):
                self._send_json(
                    {"error": "必须且只能指定 metadata 或 source 之一"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            delete_all = bool(body.get("all"))
            doc_ids = body.get("doc_ids")
            if not delete_all:
                if not isinstance(doc_ids, list) or not doc_ids:
                    self._send_json(
                        {"error": "非全量删除时须提供 doc_ids 数组"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                try:
                    doc_ids = [int(x) for x in doc_ids]
                except (TypeError, ValueError):
                    self._send_json(
                        {"error": "doc_ids 须为整数数组"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
            chunk_id = body.get("chunk_id")
            if chunk_id is not None:
                try:
                    chunk_id = int(chunk_id)
                except (TypeError, ValueError):
                    self._send_json(
                        {"error": "chunk_id 须为整数"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
            payload = {
                "metadata": metadata or None,
                "source": source or None,
                "all": delete_all,
                "doc_ids": doc_ids if not delete_all else None,
                "chunk_id": chunk_id,
            }
            ok, message = self._jobs().submit("delete", payload)
            if not ok:
                self._send_json({"error": message}, status=HTTPStatus.CONFLICT)
                return
            self._send_json({"ok": True, "message": message})
            return

        self._send_json({"error": f"unknown route: {path}"}, status=HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:
        path, parts, _ = self._parse_path()
        paths = self._paths()

        if len(parts) >= 4 and parts[0] == "api" and parts[1] == "data" and parts[2] == "files":
            relpath = "/".join(parts[3:])
            try:
                file_path = safe_path_under(paths["data_dir"], relpath)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return
            if not file_path.is_file():
                self._send_json(
                    {"error": f"file not found: {relpath}"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            file_path.unlink()
            self._send_json({"ok": True, "deleted": relpath})
            return

        self._send_json({"error": f"unknown route: {path}"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG knowledge base admin web server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host, default: 0.0.0.0")
    parser.add_argument("--port", type=int, default=17892, help="Bind port, default: 17892")
    parser.add_argument(
        "--config-path",
        default="",
        help="Optional RAG config.yaml path (default: RAG/config/config.yaml)",
    )
    parser.add_argument(
        "--html-path",
        default=str(get_default_html_path()),
        help="Path to rag_web_server.html",
    )
    parser.add_argument(
        "--js-path",
        default=str(get_default_js_path()),
        help="Path to rag_web_server.js",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path or None)
    paths = resolve_config_paths(config)
    paths["data_dir"].mkdir(parents=True, exist_ok=True)
    paths["index_dir"].mkdir(parents=True, exist_ok=True)
    paths["doc_logs_dir"].mkdir(parents=True, exist_ok=True)
    paths["visitor_state_dir"].mkdir(parents=True, exist_ok=True)

    html_path = Path(args.html_path).expanduser().resolve()
    js_path = Path(args.js_path).expanduser().resolve()

    server = ThreadingHTTPServer((args.host, args.port), RagAdminHandler)
    server.rag_config = config  # type: ignore[attr-defined]
    server.rag_paths = paths  # type: ignore[attr-defined]
    server.job_runner = JobRunner(config, paths)  # type: ignore[attr-defined]
    server.html_path = html_path  # type: ignore[attr-defined]
    server.js_path = js_path  # type: ignore[attr-defined]
    server.index_html = load_index_html(html_path)  # type: ignore[attr-defined]

    print(f"[rag-web] serving on http://{args.host}:{args.port}")
    print(f"[rag-web] data dir: {paths['data_dir']}")
    print(f"[rag-web] index dir: {paths['index_dir']}")
    print(f"[rag-web] visitor state dir: {paths['visitor_state_dir']}")
    print(f"[rag-web] html path: {html_path}")
    print(f"[rag-web] js path: {js_path}")

    def handle_sigterm(signum, frame):
        print("\n[rag-web] received SIGTERM, stopping...")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[rag-web] stopped")
    finally:
        server.server_close()
        sys.exit(0)


if __name__ == "__main__":
    main()
