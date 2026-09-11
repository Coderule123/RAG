import os
from typing import List

# 必须放在任何 import langchain_huggingface, sentence_transformers 等库之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_huggingface import HuggingFaceEmbeddings

from RAG.config.logger_runtime import get_logger

logger = get_logger("rag")

# BGE 中文检索官方查询前缀；仅用于 embed_query，入库 embed_documents 不加
DEFAULT_BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class QueryInstructionEmbeddings:
    """
    包装 HuggingFaceEmbeddings：查询加 BGE 前缀，文档编码保持原文。

    新版 HuggingFaceEmbeddings 受 Pydantic 约束，不能给实例重绑 embed_query，
    因此用组合包装，而不是改原对象字段。
    """

    def __init__(self, inner: HuggingFaceEmbeddings, instruction: str):
        self._inner = inner
        self.instruction = instruction

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        query = text or ""
        if query and not query.startswith(self.instruction):
            query = self.instruction + query
        return self._inner.embed_query(query)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class EmbeddingService:
    """
    封装 HuggingFace Embedding（新版本），供建库与查询共用。
    """

    def __init__(
        self,
        model_name: str,
        sentence_cache_dir: str,
        local_files_only: bool,
        use_query_instruction: bool = False,
        query_instruction: str = DEFAULT_BGE_ZH_QUERY_INSTRUCTION,
    ):
        logger.info("初始化 Embedding 模型: %s", model_name)
        inner = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "local_files_only": local_files_only,
            },
            encode_kwargs={"normalize_embeddings": True},
        )
        self.use_query_instruction = bool(use_query_instruction)
        self.query_instruction = (query_instruction or "").strip() or (
            DEFAULT_BGE_ZH_QUERY_INSTRUCTION
        )
        if self.use_query_instruction:
            self.model = QueryInstructionEmbeddings(inner, self.query_instruction)
            logger.info("查询向量已启用 BGE 官方前缀: %s", self.query_instruction)
        else:
            self.model = inner
            logger.info("查询向量未加 BGE 官方前缀（use_query_instruction=false）")

        logger.info("Embedding 模型初始化完成")
