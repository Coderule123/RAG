import os

# 必须放在任何 import langchain_huggingface, sentence_transformers 等库之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_huggingface import HuggingFaceEmbeddings

from RAG.config.logger_runtime import get_logger

logger = get_logger("rag")


class EmbeddingService:
    """
    封装 HuggingFace Embedding（新版本），供建库与查询共用。
    """

    def __init__(
        self, model_name: str, sentence_cache_dir: str, local_files_only: bool
    ):
        logger.info("初始化 Embedding 模型: %s", model_name)
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "local_files_only": local_files_only,
            },
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("Embedding 模型初始化完成")
