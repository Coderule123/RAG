from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS

from RAG.config.logger_runtime import get_logger

logger = get_logger("rag")

# FAISS 不支持原生 metadata 过滤，tag 过滤通过超量召回后后过滤实现。
# 该系数控制超量倍数：实际召回 top_k * TAG_FETCH_MULTIPLIER 条，过滤后截取前 top_k 条。
_TAG_FETCH_MULTIPLIER = 8


class RetrieverRuntime:
    """
    从本地 FAISS 目录加载向量库并做相似度检索。
    注意：查询必须先经与建库相同的 Embedding 编码；故 rag_api 仍须初始化 EmbeddingService。
    """

    def __init__(self, faiss_dir: str, embeddings_model):
        logger.info("加载向量库: %s", faiss_dir)
        # LangChain 在检索时会对 query 调用 embeddings_model.model.embed_query
        self.store = FAISS.load_local(
            faiss_dir,
            embeddings=embeddings_model.model,
            allow_dangerous_deserialization=True,
        )
        logger.info("向量库加载完成")

    def retrieve(
        self,
        query: str,
        top_k: int,
        tags: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        返回带 text、metadata、vector_score 的字典列表。

        tags：若不为空则只保留 metadata.tag 在列表中的结果。
        由于 FAISS 不支持原生 metadata 过滤，启用 tag 过滤时会先召回
        top_k * _TAG_FETCH_MULTIPLIER 条候选再后过滤，以保证结果充足。
        重复片段（doc_id+chunk_id 相同）的 text 字段置空。
        """
        tag_set: Optional[set] = (
            {t.lower() for t in tags if t} if tags else None
        )
        fetch_k = top_k * _TAG_FETCH_MULTIPLIER if tag_set else top_k
        logger.info(
            "开始检索: query=%s top_k=%s tags=%s fetch_k=%s",
            query, top_k, list(tag_set) if tag_set else None, fetch_k,
        )

        raw = self.store.similarity_search_with_relevance_scores(query, k=fetch_k)

        if tag_set:
            raw = [
                (doc, score)
                for doc, score in raw
                if str(doc.metadata.get("tag", "")).lower() in tag_set
            ]
            logger.info("tag 过滤后候选数: %s", len(raw))

        raw = raw[:top_k]

        results = []
        seen: set = set()
        for doc, score in raw:
            metadata = doc.metadata
            key = (metadata.get("doc_id"), metadata.get("chunk_id"))
            text = "" if key in seen else doc.page_content
            if text:
                seen.add(key)
            results.append({
                "text": text,
                "metadata": metadata,
                "vector_score": float(score),
            })

        logger.info("检索完成: hits=%s", len(results))
        return results
