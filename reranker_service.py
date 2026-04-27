from typing import Dict, List

from RAG.config.logger_runtime import get_logger

logger = get_logger("rag")


class RerankerService:
    """
    使用 sentence-transformers CrossEncoder 对召回结果按 query 相关性重排。
    """

    def __init__(
        self, model_name: str, sentence_cache_dir: str, local_files_only: bool
    ):
        logger.info("初始化 Reranker 模型: %s", model_name)
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            model_name,
            local_files_only=local_files_only,
            model_kwargs={"cache_dir": sentence_cache_dir},
        )
        logger.info("Reranker 模型初始化完成")

    def rerank(self, query: str, docs: List[Dict], top_n: int) -> List[Dict]:
        """为每条 doc 增加 rerank_score，按分数降序截取 top_n。"""
        logger.info("开始 rerank: docs=%s top_n=%s", len(docs), top_n)
        if not docs:
            return []
        pairs = [[query, d["text"]] for d in docs]
        scores = self.model.predict(pairs)
        merged = []
        for doc, score in zip(docs, scores):
            item = dict(doc)
            item["rerank_score"] = float(score)
            merged.append(item)
        merged.sort(key=lambda x: x["rerank_score"], reverse=True)
        logger.info("rerank 完成")
        return merged[:top_n]
