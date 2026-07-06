import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from RAG.config.logger_runtime import get_logger

logger = get_logger("rag")

# 精简后的销售导购流程（8 个核心阶段，按真实门店接待顺序排列）
# 每个 step 的 id 与 tour_lang_rules.py 中的规则 key 严格对应
DEFAULT_TOUR_STEPS: List[Dict[str, Any]] = [
    {
        "id": "greeting",
        "title": "① 接待问候：进店欢迎，判断是否首次来访",
        "order": 1,
    },
    {
        "id": "needs_exploration",
        "title": "② 探寻需求：用车场景、用车人、关注车型",
        "order": 2,
    },
    {
        "id": "powertrain_range",
        "title": "③ 动力续航：增程/纯电技术、800V补能、使用成本",
        "order": 3,
    },
    {
        "id": "exterior_chassis",
        "title": "④ 车外讲解：底盘/后轮转向/安全/智驾",
        "order": 4,
    },
    {
        "id": "driver_cockpit",
        "title": "⑤ 主驾体验：5K大屏、雨夜模式、一键泊车",
        "order": 5,
    },
    {
        "id": "copilot_rear",
        "title": "⑥ 副驾后排：零重力座椅、后排空间、大冰箱",
        "order": 6,
    },
    {
        "id": "test_drive",
        "title": "⑦ 邀请试驾：引导顾客实际驾驶体验",
        "order": 7,
    },
    {
        "id": "purchase_intent",
        "title": "⑧ 购买意向：价格/金融方案/下订意向确认",
        "order": 8,
    },
]

_UNSAFE_FILENAME_PATTERN = re.compile(r"[^\w\-.]")


def _safe_state_filename(vision_id: str) -> str:
    """将 vision_user_id 转为安全的单用户状态文件名。"""
    raw = (vision_id or "").strip()
    if not raw:
        return "anonymous.json"
    safe = _UNSAFE_FILENAME_PATTERN.sub("_", raw)
    if len(safe) > 120:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        safe = f"{safe[:80]}_{digest}.json"
        return safe if safe.endswith(".json") else f"{safe}.json"
    return f"{safe}.json"


def _empty_step(step_def: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": step_def["id"],
        "title": step_def.get("title", step_def["id"]),
        "order": int(step_def.get("order", 0)),
        "asked": False,
        "asked_at": None,
    }


def _default_user_state(vision_id: str, tour_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    now = time.time()
    return {
        "vision_user_id": vision_id,
        "person_name": None,
        "created_at": now,
        "updated_at": now,
        "ask_name": {"asked": False, "first_asked_at": None},
        "tour_process": {
            "current_vehicle_tag": "ls6",
            "steps": [_empty_step(step) for step in tour_steps],
        },
    }


class VisitorStateStore:
    """按 vision_user_id 独立维护用户状态文件，记录姓名询问与观车讲解进度。"""

    def __init__(
        self,
        state_dir: str,
        tour_steps: Optional[List[Dict[str, Any]]] = None,
        legacy_file: Optional[str] = None,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.tour_steps = tour_steps if tour_steps is not None else DEFAULT_TOUR_STEPS
        self._legacy_file = Path(legacy_file) if legacy_file else None
        self._migrate_legacy_file_if_needed()

    def _state_path(self, vision_id: str) -> Path:
        return self.state_dir / _safe_state_filename(vision_id)

    def _load_file(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("读取用户状态失败: path=%s err=%s", path, exc)
            return None
        return data if isinstance(data, dict) else None

    def _save_file(self, path: Path, data: Dict[str, Any]) -> None:
        data["updated_at"] = time.time()
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _normalize_state(self, vision_id: str, raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        state = _default_user_state(vision_id, self.tour_steps)
        if not raw:
            return state

        state["vision_user_id"] = str(raw.get("vision_user_id") or vision_id)
        if raw.get("person_name"):
            state["person_name"] = str(raw["person_name"])
        if isinstance(raw.get("created_at"), (int, float)):
            state["created_at"] = float(raw["created_at"])

        ask_name = raw.get("ask_name") if isinstance(raw.get("ask_name"), dict) else raw
        if isinstance(ask_name, dict):
            state["ask_name"]["asked"] = bool(ask_name.get("asked", False))
            first_asked_at = ask_name.get("first_asked_at")
            if isinstance(first_asked_at, (int, float)):
                state["ask_name"]["first_asked_at"] = float(first_asked_at)

        tour = raw.get("tour_process") if isinstance(raw.get("tour_process"), dict) else {}
        if tour.get("current_vehicle_tag"):
            state["tour_process"]["current_vehicle_tag"] = str(tour["current_vehicle_tag"])

        existing_steps = {
            step.get("id"): step
            for step in (tour.get("steps") or [])
            if isinstance(step, dict) and step.get("id")
        }
        merged_steps: List[Dict[str, Any]] = []
        for step_def in self.tour_steps:
            base = _empty_step(step_def)
            saved = existing_steps.get(step_def["id"])
            if saved:
                base["asked"] = bool(saved.get("asked", False))
                asked_at = saved.get("asked_at")
                if isinstance(asked_at, (int, float)):
                    base["asked_at"] = float(asked_at)
            merged_steps.append(base)
        state["tour_process"]["steps"] = merged_steps
        return state

    def _migrate_legacy_file_if_needed(self) -> None:
        """将旧版单文件 visitor_state.json 迁移为按用户独立文件。"""
        if self._legacy_file is None or not self._legacy_file.exists():
            return
        try:
            legacy_data = json.loads(self._legacy_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("迁移旧版访客状态失败: %s", exc)
            return
        if not isinstance(legacy_data, dict):
            return

        migrated = 0
        for vision_id, value in legacy_data.items():
            if not isinstance(vision_id, str) or not isinstance(value, dict):
                continue
            path = self._state_path(vision_id)
            if path.exists():
                continue
            state = self._normalize_state(vision_id, {
                "vision_user_id": vision_id,
                "ask_name": value,
            })
            self._save_file(path, state)
            migrated += 1

        if migrated:
            backup = self._legacy_file.with_suffix(".json.bak")
            self._legacy_file.rename(backup)
            logger.info(
                "已迁移 %d 条旧版访客状态到目录 %s，原文件备份为 %s",
                migrated,
                self.state_dir,
                backup,
            )

    def get_or_create(self, vision_id: str) -> Dict[str, Any]:
        path = self._state_path(vision_id)
        state = self._normalize_state(vision_id, self._load_file(path))
        if not path.exists():
            self._save_file(path, state)
            logger.info("新建用户状态文件: vision_user_id=%s path=%s", vision_id, path)
        return state

    def save(self, vision_id: str, state: Dict[str, Any]) -> None:
        path = self._state_path(vision_id)
        state["vision_user_id"] = vision_id
        self._save_file(path, state)

    def get(self, vision_id: str) -> Optional[Dict[str, Any]]:
        path = self._state_path(vision_id)
        raw = self._load_file(path)
        if raw is None:
            return None
        return self._normalize_state(vision_id, raw)

    def mark_asked(self, vision_id: str) -> None:
        state = self.get_or_create(vision_id)
        first_asked_at = state["ask_name"].get("first_asked_at")
        if not isinstance(first_asked_at, (int, float)):
            first_asked_at = time.time()
        state["ask_name"] = {
            "asked": True,
            "first_asked_at": float(first_asked_at),
        }
        self.save(vision_id, state)
        logger.info(
            "已记录姓名询问: vision_user_id=%s first_asked_at=%.3f",
            vision_id,
            float(first_asked_at),
        )

    def should_reask(self, vision_id: str, timeout_seconds: float) -> bool:
        """是否需要重新询问姓名。True 表示应重新询问。"""
        if timeout_seconds <= 0:
            return False

        state = self.get(vision_id)
        if state is None:
            return True

        first_asked_at = state["ask_name"].get("first_asked_at")
        if not isinstance(first_asked_at, (int, float)):
            logger.info(
                "访客状态缺少首次询问时间，补写并重新询问: vision_user_id=%s",
                vision_id,
            )
            self.mark_asked(vision_id)
            return True

        elapsed = time.time() - float(first_asked_at)
        if elapsed > float(timeout_seconds):
            logger.info(
                "访客姓名询问超时，重新询问: vision_user_id=%s elapsed=%.1fs timeout=%.1fs",
                vision_id,
                elapsed,
                float(timeout_seconds),
            )
            state = self.get_or_create(vision_id)
            state["ask_name"] = {"asked": True, "first_asked_at": float(time.time())}
            self.save(vision_id, state)
            return True
        return False

    def set_person_name(self, vision_id: str, person_name: str) -> None:
        state = self.get_or_create(vision_id)
        state["person_name"] = person_name.strip()
        self.save(vision_id, state)

    def mark_tour_step_asked(self, vision_id: str, step_id: str) -> bool:
        """标记某一观车讲解环节已询问/已讲解。返回是否成功找到并更新。"""
        state = self.get_or_create(vision_id)
        updated = False
        for step in state["tour_process"]["steps"]:
            if step["id"] == step_id and not step["asked"]:
                step["asked"] = True
                step["asked_at"] = time.time()
                updated = True
                break
        if updated:
            self.save(vision_id, state)
            logger.info(
                "已记录观车环节: vision_user_id=%s step_id=%s",
                vision_id,
                step_id,
            )
        return updated

    def get_next_pending_step(self, vision_id: str) -> Optional[Dict[str, Any]]:
        """返回下一个尚未完成的阶段（只读，不标记），供主动招呼时决定话题。"""
        state = self.get_or_create(vision_id)
        steps = sorted(state["tour_process"]["steps"], key=lambda s: s.get("order", 0))
        for step in steps:
            if not step.get("asked"):
                return step
        return None

    def mark_next_tour_step(self, vision_id: str) -> Optional[str]:
        """按顺序标记下一个尚未询问的观车环节，返回 step_id。"""
        state = self.get_or_create(vision_id)
        steps = sorted(state["tour_process"]["steps"], key=lambda s: s.get("order", 0))
        for step in steps:
            if not step.get("asked"):
                step["asked"] = True
                step["asked_at"] = time.time()
                self.save(vision_id, state)
                logger.info(
                    "已推进观车流程: vision_user_id=%s step_id=%s title=%s",
                    vision_id,
                    step["id"],
                    step.get("title"),
                )
                return step["id"]
        return None

    def get_tour_progress_summary(self, vision_id: str) -> str:
        state = self.get_or_create(vision_id)
        steps = state["tour_process"]["steps"]
        asked_count = sum(1 for step in steps if step.get("asked"))
        pending = [step["title"] for step in steps if not step.get("asked")]
        pending_preview = "、".join(pending[:3])
        if len(pending) > 3:
            pending_preview += f" 等{len(pending)}项"
        return (
            f"观车进度 {asked_count}/{len(steps)}；"
            f"待讲解：{pending_preview or '无'}"
        )

    def mark_steps_from_texts(
        self,
        vision_id: str,
        query: str = "",
        response: str = "",
    ) -> List[str]:
        """
        对 query + response 运行语言规则库，批量标记命中的观车环节。
        返回本次新标记的 step_id 列表（已标记过的不重复计入）。

        典型用法：LLM 返回回复后调用一次，传入 (query, llm_response)。
        """
        from RAG.tour_lang_rules import detect_completed_steps, steps_summary

        matched = detect_completed_steps(query=query, response=response)
        newly_marked: List[str] = []
        for step_id in matched:
            if self.mark_tour_step_asked(vision_id, step_id):
                newly_marked.append(step_id)
        if newly_marked:
            logger.info(
                "语言规则标记(含回复): vision_user_id=%s 新增=[%s]",
                vision_id,
                steps_summary(newly_marked),
            )
        return newly_marked

    def get_state_file_path(self, vision_id: str) -> str:
        return str(self._state_path(vision_id))
