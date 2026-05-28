import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from RAG.config.logger_runtime import get_logger

logger = get_logger("rag")


class VisitorStateStore:
    """访客状态存储：将视觉标识状态落盘到 JSON 文件。

    保存每个 vision_id 是否已询问（asked）以及首次询问时间（first_asked_at）。
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self._save({})

    def _load(self) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, value in data.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            asked = bool(value.get("asked", False))
            raw_first_asked_at = value.get("first_asked_at")
            first_asked_at = (
                float(raw_first_asked_at)
                if isinstance(raw_first_asked_at, (int, float))
                else None
            )
            normalized[key] = {"asked": asked, "first_asked_at": first_asked_at}
        return normalized

    def _save(self, data: Dict[str, Dict[str, Any]]) -> None:
        self.file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get(self, vision_id: str) -> Optional[Dict[str, Any]]:
        data = self._load()
        return data.get(vision_id)

    def mark_asked(self, vision_id: str) -> None:
        data = self._load()
        existing = data.get(vision_id, {})
        first_asked_at = existing.get("first_asked_at")
        if not isinstance(first_asked_at, (int, float)):
            first_asked_at = time.time()
        data[vision_id] = {"asked": True, "first_asked_at": float(first_asked_at)}
        self._save(data)

    def should_reask(self, vision_id: str, timeout_seconds: float) -> bool:
        """是否需要重新询问姓名。True 表示应重新询问。"""
        if timeout_seconds <= 0:
            return False

        state = self.get(vision_id)
        if state is None:
            return True

        first_asked_at = state.get("first_asked_at")
        if not isinstance(first_asked_at, (int, float)):
            # 老数据兼容：无时间戳时，补写当前时间并视为本次要重新询问
            logger.info(
                "访客状态缺少首次询问时间，补写并重新询问: vision_id=%s",
                vision_id,
            )
            self.mark_asked(vision_id)
            return True

        elapsed = time.time() - float(first_asked_at)
        if elapsed > float(timeout_seconds):
            logger.info(
                "访客姓名询问超时，重新询问: vision_id=%s elapsed=%.1fs timeout=%.1fs first_asked_at=%.3f",
                vision_id,
                elapsed,
                float(timeout_seconds),
                float(first_asked_at),
            )
            # 超时后把“首次询问时间”重置为当前时间，开启新周期
            data = self._load()
            data[vision_id] = {"asked": True, "first_asked_at": float(time.time())}
            self._save(data)
            return True
        return False
