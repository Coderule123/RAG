"""
文档预处理入口：加载 → 语义切分 → Embedding → 写入 FAISS（及 metadata/sqlite）。
无参数时按 config.yaml 中的 paths / vector_index / models 执行。
"""

import argparse
import json
import sys
from pathlib import Path

from RAG.config.config_runtime import load_config
from RAG.config.hf_runtime import setup_huggingface_env
from RAG.config.logger_runtime import get_logger, setup_logging
from RAG.DP.document_loader import load_documents
from RAG.DP.embedding_service import EmbeddingService
from RAG.DP.semantic_splitter import split_documents
from RAG.DP.vector_store import VectorStore

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser(config: dict) -> argparse.ArgumentParser:
    """命令行参数与 YAML 默认值对齐，便于批处理脚本覆盖。"""
    paths = config.get("paths", {})
    vector = config.get("vector_index", {})
    models = config.get("models", {})
    parser = argparse.ArgumentParser(description="文档预处理与向量索引构建")
    parser.add_argument("--data-dir", default=paths.get("data_dir", "./assets/data"))
    parser.add_argument(
        "--index-dir", default=paths.get("index_dir", "./assets/index_store")
    )
    parser.add_argument("--chunk-size", type=int, default=vector.get("chunk_size", 500))
    parser.add_argument("--overlap", type=int, default=vector.get("overlap", 80))
    parser.add_argument("--batch-size", type=int, default=vector.get("batch_size", 128))
    parser.add_argument(
        "--embedding-model",
        default=models.get("embedding_model", "BAAI/bge-small-zh-v1.5"),
    )
    parser.set_defaults(incremental=vector.get("incremental", True))
    parser.add_argument("--incremental", dest="incremental", action="store_true")
    parser.add_argument("--no-incremental", dest="incremental", action="store_false")
    return parser


def main() -> None:
    config = load_config()
    paths = config.get("paths", {})
    vector = config.get("vector_index", {})
    models = config.get("models", {})
    hf_cfg = config.get("huggingface", {})

    log_file = setup_logging(
        paths.get("doc_logs_dir", "./logs/doc"),
        logger_name="rag",
        log_mode="timestamp",
    )
    logger = get_logger("rag")
    logger.info("document_processor 启动: log_file=%s", log_file)

    hf_runtime = setup_huggingface_env(config)
    logger.info("HF 运行配置: %s", json.dumps(hf_runtime, ensure_ascii=False))

    # 无 CLI 参数时走配置文件中的默认路径与切分参数
    if len(sys.argv) == 1:
        args = argparse.Namespace(
            data_dir=paths.get("data_dir", "./assets/data"),
            index_dir=paths.get("index_dir", "./assets/index_store"),
            chunk_size=int(vector.get("chunk_size", 500)),
            overlap=int(vector.get("overlap", 80)),
            batch_size=int(vector.get("batch_size", 128)),
            incremental=bool(vector.get("incremental", True)),
            embedding_model=models.get("embedding_model", "BAAI/bge-small-zh-v1.5"),
        )
        logger.info("纯配置模式运行")
    else:
        args = build_parser(config).parse_args()
        logger.info("CLI 模式运行")

    logger.info("开始文档处理流程")
    index_dir = str(Path(args.index_dir).resolve())
    docs, load_stats = load_documents(
        args.data_dir,
        incremental=args.incremental,
        index_dir=index_dir,
    )
    if not docs:
        result = {
            "document_count": 0,
            "chunk_count": 0,
            "total_chunks": 0,
            "message": "没有新增或变更文档",
            "index_dir": index_dir,
            "doc_hash_path": str(Path(index_dir) / "doc_hash.json"),
            "metadata_docs_dir": str(Path(index_dir) / "metadata"),
            **load_stats,
        }
        logger.info("无新增文档，流程结束")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # Embedding 提前初始化：切分阶段的语义断点检测与入库共用同一模型
    embed = EmbeddingService(
        model_name=args.embedding_model,
        sentence_cache_dir=hf_runtime["sentence_cache_dir"],
        local_files_only=bool(hf_cfg.get("local_files_only", False)),
    )

    chunks = split_documents(
        docs,
        args.chunk_size,
        args.overlap,
        max_chunk_size=vector.get("max_chunk_size"),
        min_chunk_chars=int(vector.get("min_chunk_chars", 15)),
        add_context_header=bool(vector.get("add_context_header", True)),
        embeddings_model=embed,
        semantic_split=bool(vector.get("semantic_split", True)),
        semantic_breakpoint_percentile=float(
            vector.get("semantic_breakpoint_percentile", 88)
        ),
    )
    store = VectorStore(index_dir=args.index_dir, data_dir=args.data_dir)
    result = store.build_or_append(
        chunks,
        embed,
        incremental=args.incremental,
        batch_size=args.batch_size,
    )
    result.update(
        {
            "document_count": len(docs),
            "chunk_count": result.get("added_chunks", len(chunks)),
            **load_stats,
        }
    )
    logger.info("文档处理流程完成")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
