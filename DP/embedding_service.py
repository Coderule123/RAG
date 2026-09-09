import os

# 必须放在任何 import langchain_huggingface, sentence_transformers 等库之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_huggingface import HuggingFaceEmbeddings

from RAG.config.logger_runtime import get_logger

logger = get_logger("rag")

# BGE 中文检索官方查询前缀；仅用于 embed_query，入库 embed_documents 不加
DEFAULT_BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


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
        self.model = HuggingFaceEmbeddings(
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
            self._enable_query_instruction()
            logger.info("查询向量已启用 BGE 官方前缀: %s", self.query_instruction)
        else:
            logger.info("查询向量未加 BGE 官方前缀（use_query_instruction=false）")

        logger.info("Embedding 模型初始化完成")

    def _enable_query_instruction(self) -> None:
        """只包装 embed_query；文档入库走 embed_documents，不受前缀影响。"""
        orig_embed_query = self.model.embed_query
        prefix = self.query_instruction

        def embed_query_with_instruction(text: str):
            query = text or ""
            if query and not query.startswith(prefix):
                query = prefix + query
            return orig_embed_query(query)

        self.model.embed_query = embed_query_with_instruction
