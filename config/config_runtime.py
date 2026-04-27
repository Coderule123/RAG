"""从 YAML 加载项目配置，供 DP 与 RAG 脚本共用。"""

from pathlib import Path
from typing import Any

import yaml

from RAG.config.logger_runtime import get_logger

logger = get_logger("config")


def get_project_root() -> Path:
    """项目根目录：包含 config/、DP/、RAG/ 的那一层（即 RAG/RAG）。"""
    return Path(__file__).resolve().parent.parent


def get_default_config_path() -> Path:
    """默认配置文件路径（与 get_project_root 下的 config/config.yaml 对应）。"""
    return get_project_root() / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    读取 YAML；文件缺失或解析失败时打日志并返回空 dict，避免调用方 KeyError 连锁崩溃。
    """
    path = Path(config_path) if config_path else get_default_config_path()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.error("加载配置文件失败: %s", exc)
        return {}
