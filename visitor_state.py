import json
from pathlib import Path
from typing import Any, Dict, Optional


class VisitorStateStore:
    """访客状态存储：将视觉标识状态落盘到 JSON 文件。

    现在仅保存每个 vision_id 是否已询问（asked）。
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
            normalized[key] = {"asked": bool(value.get("asked", False))}
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
        data[vision_id] = {"asked": True}
        self._save(data)
