from typing import Dict, List

from langchain_community.vectorstores import FAISS

from RAG.config.logger_runtime import get_logger

logger = get_logger("rag")


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

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        """返回带 text、metadata、vector_score 的字典列表（分数依 LangChain 实现）。重复片段（doc_id+chunk_id相同）的 text 字段设为 'NONE'。"""
        logger.info("开始检索: query=%s top_k=%s", query, top_k)
        docs = self.store.similarity_search_with_relevance_scores(query, k=top_k)

        results = []
        seen = set()  # 记录已出现过的 (doc_id, chunk_id)

        for doc, score in docs:
            metadata = doc.metadata
            doc_id = metadata.get('doc_id')
            chunk_id = metadata.get('chunk_id')
            key = (doc_id, chunk_id)  # 唯一标识片段

            if key in seen:
                # 重复片段为空
                text = ""
            else:
                text = doc.page_content
                seen.add(key)

            results.append({
                "text": text,
                "metadata": metadata,
                "vector_score": float(score),
            })

        logger.info("检索完成: hits=%s", len(results))
        return results
