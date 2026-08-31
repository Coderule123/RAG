import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from RAG.config.logger_runtime import get_logger

logger = get_logger("rag")

# 顾问式汽车展厅销售流程（9 个核心阶段，按真实接待顺序排列）
# 每个 step 的 id 与 tour_lang_rules.py 中的规则 key 严格对应
#
# 设计原则：
#   - 以展厅机器人实际能参与的环节为边界，不含交车/售后等离店流程
#   - greeting（破冰）与 interest_probe（意向摸底）分离，避免初次接触就强推需求问题
#   - contact_retention 作为收尾，聚焦"留联系方式/预约再访"，是展厅机器人最高价值动作
DEFAULT_TOUR_STEPS: List[Dict[str, Any]] = [
    {
        "id": "greeting",
        "title": "① 展厅破冰：进店问候、建立信任、判断首次来访",
        "order": 1,
    },
    {
        "id": "interest_probe",
        "title": "② 意向摸底：了解来意、粗粒度兴趣方向、购车成熟度",
        "order": 2,
    },
    {
        "id": "needs_analysis",
        "title": "③ 深度需求：用途场景、预算区间、决策人、换购原因",
        "order": 3,
    },
    {
        "id": "vehicle_selection",
        "title": "④ 车型推荐：匹配车型、版本和配置方向",
        "order": 4,
    },
    {
        "id": "product_presentation",
        "title": "⑤ 车辆展示：六方位讲解与核心卖点体验",
        "order": 5,
    },
    {
        "id": "test_drive",
        "title": "⑥ 试乘试驾：邀约试驾、路线说明、体验反馈",
        "order": 6,
    },
    {
        "id": "quote_negotiation",
        "title": "⑦ 报价协商：价格、权益、金融、置换方案",
        "order": 7,
    },
    {
        "id": "deal_confirmation",
        "title": "⑧ 成交确认：配置颜色、下订意向、异议处理",
        "order": 8,
    },
    {
        "id": "contact_retention",
        "title": "⑨ 留档跟进：留联系方式、预约回访、邀请关注",
        "order": 9,
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


def _empty_interest_profile() -> Dict[str, Any]:
    """用户喜好感知：按车型记录询问过的关注点。"""
    return {
        "vehicles": {},
        # 最近一次明确提到的车型，供后续「只问关注点、未提车型」时归属
        "last_vehicle_tag": None,
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
            "steps": [_empty_step(step) for step in tour_steps],
        },
        "interest_profile": _empty_interest_profile(),
    }


class VisitorStateStore:
    """按 vision_user_id 独立维护用户状态：姓名、导购阶段进度、车型喜好感知。"""

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
        state["interest_profile"] = self._normalize_interest_profile(
            raw.get("interest_profile")
        )
        return state

    @staticmethod
    def _normalize_interest_profile(raw: Any) -> Dict[str, Any]:
        profile = _empty_interest_profile()
        if not isinstance(raw, dict):
            return profile

        last_tag = raw.get("last_vehicle_tag")
        if isinstance(last_tag, str) and last_tag.strip():
            profile["last_vehicle_tag"] = last_tag.strip().lower()

        vehicles_raw = raw.get("vehicles")
        if not isinstance(vehicles_raw, dict):
            return profile

        vehicles: Dict[str, Any] = {}
        for tag, info in vehicles_raw.items():
            if not isinstance(tag, str) or not tag.strip():
                continue
            if not isinstance(info, dict):
                continue
            norm_tag = tag.strip().lower()
            topics_out: Dict[str, Any] = {}
            topics_raw = info.get("topics") if isinstance(info.get("topics"), dict) else {}
            for topic_id, topic_info in topics_raw.items():
                if not isinstance(topic_id, str) or not topic_id.strip():
                    continue
                if not isinstance(topic_info, dict):
                    continue
                tid = topic_id.strip()
                count = topic_info.get("count", 1)
                try:
                    count_i = max(1, int(count))
                except (TypeError, ValueError):
                    count_i = 1
                entry = {
                    "title": str(topic_info.get("title") or tid),
                    "count": count_i,
                    "first_asked_at": None,
                    "last_asked_at": None,
                }
                for key in ("first_asked_at", "last_asked_at"):
                    val = topic_info.get(key)
                    if isinstance(val, (int, float)):
                        entry[key] = float(val)
                topics_out[tid] = entry

            vehicle_entry = {
                "ask_count": 1,
                "first_asked_at": None,
                "last_asked_at": None,
                "topics": topics_out,
            }
            try:
                vehicle_entry["ask_count"] = max(1, int(info.get("ask_count", 1)))
            except (TypeError, ValueError):
                vehicle_entry["ask_count"] = 1
            for key in ("first_asked_at", "last_asked_at"):
                val = info.get(key)
                if isinstance(val, (int, float)):
                    vehicle_entry[key] = float(val)
            vehicles[norm_tag] = vehicle_entry

        profile["vehicles"] = vehicles
        if profile["last_vehicle_tag"] is None and vehicles:
            # 无 last 标记时，取最近一次询问的车型
            newest = max(
                vehicles.items(),
                key=lambda item: float(item[1].get("last_asked_at") or 0),
            )
            profile["last_vehicle_tag"] = newest[0]
        return profile

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
        raw = self._load_file(path)
        state = self._normalize_state(vision_id, raw)
        if not path.exists():
            self._save_file(path, state)
            logger.info("新建用户状态文件: vision_user_id=%s path=%s", vision_id, path)
        else:
            # 兼容迁移：去掉已废弃的 current_vehicle_tag，并补齐 interest_profile
            tour_raw = raw.get("tour_process") if isinstance(raw, dict) else None
            needs_migrate = (
                isinstance(tour_raw, dict) and "current_vehicle_tag" in tour_raw
            ) or (isinstance(raw, dict) and "interest_profile" not in raw)
            if needs_migrate:
                self._save_file(path, state)
                logger.info(
                    "已迁移用户状态结构: vision_user_id=%s path=%s",
                    vision_id,
                    path,
                )
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
        """返回下一个待推进的阶段（只读，不标记），供主动招呼时决定话题。

        策略：以"已完成的最高 order"为下限向后寻找，防止因规则乱序触发导致主动
        对话倒退到早已经历过的阶段。若后续全部完成，则返回 None；若没有任何已完成
        阶段（全新访客），则从头开始。
        """
        state = self.get_or_create(vision_id)
        steps = sorted(state["tour_process"]["steps"], key=lambda s: s.get("order", 0))

        # 找到已完成阶段的最高 order（0 表示尚无任何已完成阶段）
        max_done_order = max(
            (s.get("order", 0) for s in steps if s.get("asked")),
            default=0,
        )

        # 优先：从最高已完成 order 之后找第一个未完成阶段
        for step in steps:
            if not step.get("asked") and step.get("order", 0) > max_done_order:
                return step

        # 兜底：若跳跃式触发导致前序有遗漏，返回最早的未完成阶段（保持对话完整性）
        for step in steps:
            if not step.get("asked"):
                return step

        return None

    def get_active_ask_context(self, vision_id: str) -> Dict[str, Any]:
        """供主动招呼 prompt 使用的结构化进展（只读，不推进阶段）。"""
        state = self.get_or_create(vision_id)
        next_step = self.get_next_pending_step(vision_id)
        steps = sorted(state["tour_process"]["steps"], key=lambda s: s.get("order", 0))
        done = [str(step.get("title") or step["id"]) for step in steps if step.get("asked")]
        pending = [str(step.get("title") or step["id"]) for step in steps if not step.get("asked")]
        person_name = str(state.get("person_name") or "").strip()
        return {
            "person_name": person_name,
            "done_titles": done,
            "pending_titles": pending,
            "current_step_id": str(next_step["id"]) if next_step else "",
            "current_step_title": (
                str(next_step.get("title") or next_step["id"]) if next_step else ""
            ),
            "interest_summary": self.get_interest_summary(vision_id),
            "is_new_visitor": len(done) == 0,
        }

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
        vehicle_tags: Optional[List[str]] = None,
    ) -> List[str]:
        """
        对 query + response 运行语言规则库，批量标记命中的观车环节，
        并同步更新用户喜好感知（询问过的车型与关注点）。
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
        self.record_preferences_from_texts(
            vision_id,
            query=query,
            response=response,
            vehicle_tags=vehicle_tags,
        )
        return newly_marked

    def get_last_vehicle_tag(self, vision_id: str) -> Optional[str]:
        state = self.get_or_create(vision_id)
        profile = state.get("interest_profile") or {}
        last = profile.get("last_vehicle_tag")
        return str(last).strip().lower() if last else None

    def record_preferences_from_texts(
        self,
        vision_id: str,
        query: str = "",
        response: str = "",
        vehicle_tags: Optional[List[str]] = None,
        fallback_vehicles: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        根据本轮对话更新喜好感知：记录询问过的车型，以及每个车型下的关注点。
        返回本轮识别结果 {"vehicles": [...], "topics": [...]}。
        """
        from RAG.preference_rules import (
            detect_preferences,
            preferences_summary,
            topic_title,
        )

        state = self.get_or_create(vision_id)
        profile = state.setdefault("interest_profile", _empty_interest_profile())
        if not isinstance(profile.get("vehicles"), dict):
            profile["vehicles"] = {}

        fallback = list(fallback_vehicles or [])
        last_tag = profile.get("last_vehicle_tag")
        if last_tag and last_tag not in fallback:
            fallback.append(str(last_tag))

        detected = detect_preferences(
            query=query,
            response=response,
            vehicle_tags=vehicle_tags,
            fallback_vehicles=fallback,
        )
        vehicles = detected.get("vehicles") or []
        topics = detected.get("topics") or []
        if not vehicles and not topics:
            return detected

        now = time.time()
        vehicles_map: Dict[str, Any] = profile["vehicles"]
        for tag in vehicles:
            entry = vehicles_map.get(tag)
            if not isinstance(entry, dict):
                entry = {
                    "ask_count": 0,
                    "first_asked_at": now,
                    "last_asked_at": now,
                    "topics": {},
                }
                vehicles_map[tag] = entry
            entry["ask_count"] = int(entry.get("ask_count") or 0) + 1
            if not isinstance(entry.get("first_asked_at"), (int, float)):
                entry["first_asked_at"] = now
            entry["last_asked_at"] = now
            if not isinstance(entry.get("topics"), dict):
                entry["topics"] = {}

            for topic_id in topics:
                topic_entry = entry["topics"].get(topic_id)
                if not isinstance(topic_entry, dict):
                    topic_entry = {
                        "title": topic_title(topic_id),
                        "count": 0,
                        "first_asked_at": now,
                        "last_asked_at": now,
                    }
                    entry["topics"][topic_id] = topic_entry
                topic_entry["title"] = topic_title(topic_id)
                topic_entry["count"] = int(topic_entry.get("count") or 0) + 1
                if not isinstance(topic_entry.get("first_asked_at"), (int, float)):
                    topic_entry["first_asked_at"] = now
                topic_entry["last_asked_at"] = now

            profile["last_vehicle_tag"] = tag

        # 仅命中关注点、无车型且无 fallback 时，暂不写入，避免无法归属
        self.save(vision_id, state)
        logger.info(
            "喜好感知更新: vision_user_id=%s %s",
            vision_id,
            preferences_summary(vehicles, topics),
        )
        return detected

    def get_interest_summary(self, vision_id: str) -> str:
        """生成可读的喜好摘要，供日志/prompt 使用。"""
        state = self.get_or_create(vision_id)
        profile = state.get("interest_profile") or {}
        vehicles = profile.get("vehicles") if isinstance(profile, dict) else {}
        if not isinstance(vehicles, dict) or not vehicles:
            return "喜好感知：暂无车型关注记录"

        parts: List[str] = []
        for tag, info in vehicles.items():
            if not isinstance(info, dict):
                continue
            topics = info.get("topics") if isinstance(info.get("topics"), dict) else {}
            titles = [
                str(t.get("title") or tid)
                for tid, t in topics.items()
                if isinstance(t, dict)
            ]
            if titles:
                parts.append(f"{tag}({ '、'.join(titles) })")
            else:
                parts.append(str(tag))
        return "喜好感知：已关注 " + "；".join(parts)

    def get_state_file_path(self, vision_id: str) -> str:
        return str(self._state_path(vision_id))
