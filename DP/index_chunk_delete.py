"""
按 metadata 分文件 JSON 或原始 source 路径，从 FAISS 中删除指定向量，并重写 metadata/、doc_hash.json、chunks.db。

须使用与建库相同的 Embedding 模型（见 config/config.yaml 的 models.embedding_model）。
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from RAG.config.config_runtime import load_config
from RAG.config.hf_runtime import setup_huggingface_env
from RAG.config.logger_runtime import get_logger, setup_logging
from RAG.DP.embedding_service import EmbeddingService
from RAG.DP.vector_store import VectorStore


def build_parser(config: dict) -> argparse.ArgumentParser:
    paths = config.get("paths", {})
    models = config.get("models", {})

    parser = argparse.ArgumentParser(
        description="按 doc_id / chunk_id 或全量删除某文档在向量库中的 chunk，并同步 metadata 与 sqlite"
    )
    parser.add_argument(
        "--index-dir",
        default=paths.get("index_dir", "./RAG/assets/index_store"),
        help="向量库根目录（含 faiss_store、metadata、doc_hash.json、chunks.db）",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--metadata",
        metavar="NAME",
        help="metadata 目录下 JSON 文件名，如 car_info.json（与入库后导出的文件名一致）",
    )
    src.add_argument(
        "--source",
        metavar="PATH",
        help="数据文件路径（与 chunk 中 metadata.source 一致，如 RAG/assets/data/car_info.json）",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--all",
        action="store_true",
        help="删除该来源在索引中的全部向量",
    )
    mode.add_argument(
        "--doc-id",
        dest="doc_ids",
        action="append",
        type=int,
        help="要删除的 doc_id，可重复传入多个；与 semantic_splitter 写入的 doc_id 一致",
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        default=None,
        help="可选；指定时仅删除匹配 doc_id 且 chunk_id 等于该值的向量",
    )
    parser.add_argument(
        "--embedding-model",
        default=models.get("embedding_model", "BAAI/bge-small-zh-v1.5"),
        help="须与建库时一致",
    )
    return parser


def main() -> None:
    config = load_config()
    paths = config.get("paths", {})
    hf_cfg = config.get("huggingface", {})

    hf_runtime = setup_huggingface_env(config)
    args = build_parser(config).parse_args()

    log_file = setup_logging(paths.get("doc_logs_dir", "./RAG/logs/doc"), logger_name="rag")
    logger = get_logger("rag")
    logger.info("index_chunk_delete 启动: log_file=%s", log_file)

    index_dir = str(Path(args.index_dir).resolve())
    embed = EmbeddingService(
        model_name=args.embedding_model,
        sentence_cache_dir=hf_runtime["sentence_cache_dir"],
        local_files_only=bool(hf_cfg.get("local_files_only", False)),
    )
    store = VectorStore(index_dir=index_dir)
    result = store.delete_by_metadata_selector(
        embed,
        source=args.source,
        metadata_json=args.metadata,
        delete_all=bool(args.all),
        doc_ids=args.doc_ids,
        chunk_id=args.chunk_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
