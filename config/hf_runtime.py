"""在进程内设置 Hugging Face / Transformers / SentenceTransformers 相关环境变量。"""
import os
from pathlib import Path
from typing import Any


def setup_huggingface_env(config: dict[str, Any]) -> dict[str, str]:
    """
    根据 config 的 paths.models_dir 与 huggingface 段，统一缓存目录与镜像端点。
    offline_mode 为真时设置 HF_HUB_OFFLINE，禁止联网拉模型。
    返回值供日志打印，便于排查路径是否指向预期磁盘。
    """
    paths = config.get("paths", {})
    hf_cfg = config.get("huggingface", {})
    models_dir = Path(paths.get("models_dir", "./assets/models")).resolve()
    hf_home = models_dir / "hf_home"
    hub_cache = models_dir / "hub"
    transformers_cache = models_dir / "transformers"
    sentence_cache = models_dir / "sentence_transformers"
    for p in [models_dir, hf_home, hub_cache, transformers_cache, sentence_cache]:
        p.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(sentence_cache)

    if hf_cfg.get("mirror_enabled", True):
        endpoint = hf_cfg.get("endpoint", "https://hf-mirror.com")
        os.environ["HF_ENDPOINT"] = endpoint
        os.environ["HF_HUB_BASE_URL"] = endpoint
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint

    offline_mode = bool(hf_cfg.get("offline_mode", False))
    if offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    return {
        "models_dir": str(models_dir),
        "sentence_cache_dir": str(sentence_cache),
        "local_files_only": str(bool(hf_cfg.get("local_files_only", False))),
        "offline_mode": str(offline_mode),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
    }
