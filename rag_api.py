"""
RAG 运行时 CLI：检索 → 重排 → 上下文 → 完整 Prompt（JSON 输出）。
不启动 HTTP、不调 LLM。

关于是否加载 Embedding：
FAISS 向量检索需要把用户 query 编码为与建库时同一向量空间的向量；
LangChain 的 FAISS.load_local / similarity_search 依赖传入的 Embeddings 对象完成 query 编码，
因此本脚本仍需加载 Embedding 模型（与 document_processor 使用同一 embedding_model 配置）。
若将来改为关键词检索或预计算 query 向量等方案，可再剥离运行时 Embedding。
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from RAG.config.config_runtime import load_config
from RAG.config.hf_runtime import setup_huggingface_env
from RAG.config.logger_runtime import get_logger, setup_logging
from RAG.DP.embedding_service import EmbeddingService

from .context_builder import build_context
from .prompt_builder import build_prompt
from .reranker_service import RerankerService
from .retriever_runtime import RetrieverRuntime

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser(config: dict) -> argparse.ArgumentParser:
    """解析命令行；未指定的数值回退到 config.yaml 的 retrieval 段。"""
    retrieval = config.get("retrieval", {})
    parser = argparse.ArgumentParser(
        description="RAG 运行时：检索、重排、上下文与提示词生成"
    )
    parser.add_argument(
        "--query", required=False, help="用户问题（不提供则进入交互模式）"
    )
    parser.add_argument("--top-k", type=int, default=int(retrieval.get("top_k", 8)))
    parser.add_argument(
        "--rerank-top-n", type=int, default=int(retrieval.get("rerank_top_n", 4))
    )
    return parser


CONFIG = load_config()
PATHS = CONFIG.get("paths", {})
MODELS = CONFIG.get("models", {})
RETRIEVAL = CONFIG.get("retrieval", {})
HF_CFG = CONFIG.get("huggingface", {})

LOG_FILE = setup_logging(PATHS.get("logs_dir", "./logs"), logger_name="rag")
logger = get_logger("rag")
logger.info("rag_runtime 启动: log_file=%s", LOG_FILE)
HF_RUNTIME = setup_huggingface_env(CONFIG)
logger.info("HF 运行配置: %s", json.dumps(HF_RUNTIME, ensure_ascii=False))


class RAGService:
    """RAG 服务封装：初始化一次，可多次调用 query() 进行检索与提示词生成。"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化 RAG 服务，加载 Embedding 模型、FAISS 索引、重排模型。
        :param config: 配置字典，若为 None 则使用全局 CONFIG。
        """
        self.config = config if config is not None else CONFIG
        paths = self.config.get("paths", {})
        print("RAGService 配置路径: ", paths)

        models = self.config.get("models", {})
        print("RAGService 配置模型: ", models)

        hf_cfg = self.config.get("huggingface", {})
        print("RAGService 配置 HuggingFace: ", hf_cfg)

        hf_runtime = setup_huggingface_env(self.config)
        print("RAGService HF 运行配置: ", hf_runtime)

        # 读取重排开关
        retrieval_cfg = self.config.get("retrieval", {})
        self.use_reranker = retrieval_cfg.get("use_reranker", True)

        # 初始化 Embedding 服务（与建库同模型）
        self.embed_service = EmbeddingService(
            model_name=models.get("embedding_model", "BAAI/bge-small-zh-v1.5"),
            sentence_cache_dir=hf_runtime["sentence_cache_dir"],
            local_files_only=bool(hf_cfg.get("local_files_only", False)),
        )

        # 初始化检索器（加载 FAISS 索引）
        index_dir = Path(paths.get("index_dir", "./assets/index_store")) / "faiss_store"
        self.retriever = RetrieverRuntime(
            faiss_dir=str(index_dir),
            embeddings_model=self.embed_service,
        )

        # 初始化重排器
        if self.use_reranker:
            self.reranker = RerankerService(
                model_name=models.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
                sentence_cache_dir=hf_runtime["sentence_cache_dir"],
                local_files_only=bool(hf_cfg.get("local_files_only", False)),
            )
        else:
            self.reranker = None
            logger.info("重排已禁用（use_reranker=false），将跳过重排步骤")

        logger.info("RAGService 初始化完成（模型已加载）")

    def _print_result(self, result: Dict[str, Any]) -> None:
        """打印 RAG 结果的详细信息（日志+控制台），与原有 print_rag_result 功能一致"""
        logger.info("检索耗时: %.4fs", result["timings"]["retrieve"])
        logger.info("重排耗时: %.4fs", result["timings"]["rerank"])
        logger.info(
            "上下文与提示词构建耗时: %.4fs", result["timings"]["build_context_prompt"]
        )
        logger.info("总耗时(不含LLM): %.4fs", result["timings"]["total"])

        output = {
            "query": result["query"],
            "retrieved_count": len(result["retrieved_docs"]),
            "reranked_count": len(result["reranked_docs"]),
        }
        logger.info("RAG 运行时输出:\n%s", json.dumps(output, ensure_ascii=False, indent=2))

        logger.info("=" * 60)
        logger.info("context一览:")
        for idx, doc in enumerate(result["reranked_docs"], 1):
            logger.info(f"片段 {idx}:")
            logger.info(doc.get("text", ""))
            logger.info(f"元数据: {doc.get('metadata', {})}")
            logger.info(
                f"向量分: {doc.get('vector_score')}, 重排分: {doc.get('rerank_score')}"
            )
        logger.info("-" * 40)

        logger.info("=" * 60)
        logger.info("生成的 Prompt（供 LLM 使用）:")
        logger.info(result["prompt"])
        logger.info("=" * 60)

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """执行完整的 RAG 流程：检索 → 重排 → 构建上下文 → 生成 Prompt。"""
        if not query or not query.strip():
            raise ValueError("query 不能为空")
        query = query.strip()

        # 参数默认值
        retrieval_cfg = self.config.get("retrieval", {})
        vector_threshold = retrieval_cfg.get("vector_threshold", 0.6)
        top_k = top_k if top_k is not None else int(retrieval_cfg.get("top_k", 8))
        rerank_top_n = (
            rerank_top_n
            if rerank_top_n is not None
            else int(retrieval_cfg.get("rerank_top_n", 4))
        )

        timings = {}

        # 检索
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(query, top_k)
        timings["retrieve"] = time.perf_counter() - t0

        # 重排（根据配置决定是否执行）
        if self.use_reranker and self.reranker is not None:
            t1 = time.perf_counter()
            reranked = self.reranker.rerank(query, retrieved, rerank_top_n)
            timings["rerank"] = time.perf_counter() - t1
        else:
            # 未启用重排时，直接使用检索结果，并截取前 rerank_top_n 条
            reranked = retrieved[:rerank_top_n]
            timings["rerank"] = 0.0

        # 构建上下文和 Prompt
        t2 = time.perf_counter()
        context = build_context(reranked, vector_threshold=vector_threshold)
        prompt = build_prompt(query, context)
        timings["build_context_prompt"] = time.perf_counter() - t2
        timings["total"] = (
            timings["retrieve"] + timings["rerank"] + timings["build_context_prompt"]
        )

        result = {
            "query": query,
            "retrieved_docs": retrieved,
            "reranked_docs": reranked,
            "context": context,
            "prompt": prompt,
            "timings": timings,
        }
        if verbose:
            self._print_result(result)

        return result


def print_rag_result(result: Dict[str, Any]) -> None:
    """打印 RAG 结果的详细信息（日志+控制台）。"""
    logger.info("检索耗时: %.4fs", result["timings"]["retrieve"])
    logger.info("重排耗时: %.4fs", result["timings"]["rerank"])
    logger.info(
        "上下文与提示词构建耗时: %.4fs", result["timings"]["build_context_prompt"]
    )
    logger.info("总耗时(不含LLM): %.4fs", result["timings"]["total"])

    output = {
        "query": result["query"],
        "retrieved_count": len(result["retrieved_docs"]),
        "reranked_count": len(result["reranked_docs"]),
    }
    logger.info("RAG 运行时输出:\n%s", json.dumps(output, ensure_ascii=False, indent=2))

    logger.info("=" * 60)
    logger.info("context一览:")
    for idx, doc in enumerate(result["reranked_docs"], 1):
        logger.info(f"片段 {idx}:")
        logger.info(doc.get("text", ""))
        logger.info(f"元数据: {doc.get('metadata', {})}")
        logger.info(
            f"向量分: {doc.get('vector_score')}, 重排分: {doc.get('rerank_score')}"
        )
    logger.info("-" * 40)

    logger.info("=" * 60)
    logger.info("生成的 Prompt（供 LLM 使用）:")
    logger.info(result["prompt"])
    logger.info("=" * 60)


def interactive_mode(rag_service: RAGService, top_k: int, rerank_top_n: int) -> None:
    """交互式模式：循环读取用户输入，输出 RAG 结果，输入 exit 退出。"""
    print("\n进入 RAG 交互模式，输入问题后按回车查看结果，输入 exit 或 quit 退出。")
    while True:
        try:
            user_input = input("\n请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出交互模式")
            break

        if user_input.lower() in ("exit", "quit"):
            print("退出交互模式")
            break

        if not user_input:
            print("问题不能为空，请重新输入。")
            continue

        try:
            rag_service.query(
                user_input, top_k=top_k, rerank_top_n=rerank_top_n
            )
        except Exception as e:
            print(f"处理问题时出错: {e}")
            logger.exception("交互模式查询异常")


def main() -> None:
    logger.info("启动RAG服务中...")
    args = build_parser(CONFIG).parse_args()

    # 创建服务（单次使用，脚本结束后销毁）
    rag_service = RAGService(config=CONFIG)
    logger.info("RAG服务启动完成")

    if args.query is not None:
        # 单次查询模式
        logger.info("开始 RAG 流程（检索->重排->上下文->Prompt）")
        query = args.query.strip()
        if not query:
            raise ValueError("query 不能为空")

        rag_service.query(
            query=query,
            top_k=args.top_k,
            rerank_top_n=args.rerank_top_n,
        )
        logger.info("完成一次 RAG 服务流程")
    else:
        # 交互模式
        interactive_mode(rag_service, args.top_k, args.rerank_top_n)


if __name__ == "__main__":
    main()
