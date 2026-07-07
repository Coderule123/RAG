"""
RAG 运行时 CLI：检索 → 重排 → 上下文 → 完整 Prompt（JSON 输出）。
不启动 HTTP、不调 LLM。

关于是否加载 Embedding：
FAISS 向量检索需要把用户 query 编码为与建库时同一向量空间的向量；
LangChain 的 FAISS.load_local / similarity_search 依赖传入的 Embeddings 对象完成 query 编码，
因此本脚本仍需加载 Embedding 模型（与 document_processor 使用同一 embedding_model 配置）。
若将来改为关键词检索或预计算 query 向量等方案，可再剥离运行时 Embedding。
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from RAG.config.config_runtime import load_config
from RAG.config.hf_runtime import setup_huggingface_env
from RAG.config.logger_runtime import get_logger, setup_logging
from RAG.DP.embedding_service import EmbeddingService

from .context_builder import build_context
from .prompt_builder import (
    ACTIVE_ASK_TAG,
    DEFAULT_ACTIVE_ASK_RETRIEVAL_QUERY,
    DEFAULT_GREETING_LOCATION_LABEL,
    GREETING_LOCATION_TAG,
    MOVE_TO_WAIT_INTENT_PREFIX,
    build_prompt,
    is_concrete_person_name,
    resolve_pending_navigation_tags,
)
from .reranker_service import RerankerService
from .retriever_runtime import RetrieverRuntime
from .tour_lang_rules import detect_completed_steps, steps_summary
from .visitor_state import VisitorStateStore

# ── 主动招呼阶段感知配置 ──────────────────────────────────────────────────────
# step_id -> (检索用 tag, 检索语义词, 情境提示传给 LLM)
# tag 与 data/ 下对应子目录名严格一致（DP 建库时按目录名打 tag）
STAGE_ACTIVE_ASK_CONFIG: dict = {
    "greeting": (
        "active_ask_greeting",
        "进店问候 欢迎顾客 第一次来店 首次拜访 展厅接待",
        "顾客刚进入展厅，尚未开口，请主动问候并判断是否首次来访",
    ),
    "needs_exploration": (
        "active_ask_needs",
        "探寻需求 购车意向 用车场景 家庭用车 关注哪款车",
        "已完成问候，需主动了解顾客用车场景、家庭情况与关注车型",
    ),
    "powertrain_range": (
        "active_ask_power",
        "续航介绍 增程技术 充电 使用成本 引导讲解动力",
        "需求已了解，主动引导顾客关注动力续航与使用成本亮点",
    ),
    "exterior_chassis": (
        "active_ask_exterior",
        "车外讲解 底盘 后轮转向 安全 引导走到展车旁边",
        "动力讲完，带顾客走到展车旁，主动引导观察底盘与安全配置",
    ),
    "driver_cockpit": (
        "active_ask_cockpit",
        "引导上车 主驾体验 大屏幕 座舱介绍 雨夜模式",
        "车外讲完，引导顾客坐进主驾，主动介绍座舱与智能功能",
    ),
    "copilot_rear": (
        "active_ask_rear",
        "副驾 后排空间 冰箱 零重力 贵妃椅 后备箱",
        "主驾体验完毕，带顾客移步副驾和后排，主动介绍舒适与家用配置",
    ),
    "test_drive": (
        "active_ask_testdrive",
        "邀请试驾 安排试驾 亲自体验驾驶 感受上路",
        "产品介绍已完成，主动邀请顾客安排试驾体验",
    ),
    "purchase_intent": (
        "active_ask_purchase",
        "价格 优惠 金融方案 下订 购买意向 交付周期",
        "试驾结束，主动引导顾客进入价格与购买意向沟通",
    ),
}
# ─────────────────────────────────────────────────────────────────────────────

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser(config: dict) -> argparse.ArgumentParser:
    """解析命令行；未指定的数值回退到 config.yaml 的 retrieval 段。"""
    retrieval = config.get("retrieval", {})
    parser = argparse.ArgumentParser(
        description="RAG 运行时：检索、重排、上下文与提示词生成"
    )
    parser.add_argument(
        "--query", required=False, help="用户问题（不提供则进入交互模式）"
    )
    parser.add_argument("--top-k", type=int, default=int(retrieval.get("top_k", 8)))
    parser.add_argument(
        "--rerank-top-n", type=int, default=int(retrieval.get("rerank_top_n", 4))
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="当前用户姓名（可选）；也可在问题末尾使用「  --姓名」，例如：RAG是什么？ --张三",
    )
    return parser


CONFIG = load_config()
PATHS = CONFIG.get("paths", {})
MODELS = CONFIG.get("models", {})
RETRIEVAL = CONFIG.get("retrieval", {})
HF_CFG = CONFIG.get("huggingface", {})

LOG_FILE = setup_logging(PATHS.get("logs_dir", "./logs"), logger_name="rag")
logger = get_logger("rag")
logger.info("rag_runtime 启动: log_file=%s", LOG_FILE)
HF_RUNTIME = setup_huggingface_env(CONFIG)
logger.info("HF 运行配置: %s", json.dumps(HF_RUNTIME, ensure_ascii=False))

class RAGService:
    """RAG 服务封装：初始化一次，可多次调用 query() 进行检索与提示词生成。"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化 RAG 服务，加载 Embedding 模型、FAISS 索引、重排模型。
        :param config: 配置字典，若为 None 则使用全局 CONFIG。
        """
        self.config = config if config is not None else CONFIG
        paths = self.config.get("paths", {})
        print("RAGService 配置路径: ", paths)

        models = self.config.get("models", {})
        print("RAGService 配置模型: ", models)

        hf_cfg = self.config.get("huggingface", {})
        print("RAGService 配置 HuggingFace: ", hf_cfg)

        hf_runtime = setup_huggingface_env(self.config)
        print("RAGService HF 运行配置: ", hf_runtime)

        state_dir = paths.get("visitor_state_dir", "./RAG/assets/visitor_states")
        legacy_state_file = paths.get(
            "visitor_state_file", "./RAG/assets/visitor_state.json"
        )
        self.visitor_state = VisitorStateStore(
            state_dir=state_dir,
            legacy_file=legacy_state_file,
        )
        logger.info("用户状态目录: %s", state_dir)

        # 每个实例维护自己的 second 标志，初始化为 False
        self.second = False
        self.second_ask = False
        # 触发 second_ask 时对应的 visitor id，用于检测 user_id 切换
        self._pending_name_user_id: Optional[str] = None

        # tag 检索状态：_active_tags 跨 query() 调用持久保存，实现「沿用上次 tag」
        self._active_tags: List[str] = []
        # 参观意向固定 tag：输出 LOCATION 时写入，下次参观意向前不被 query 解析覆盖，仅叠加
        self._pinned_navigation_tags: List[str] = []
        # 机器人当前所处位置（展车 tag 或 greeting=门口迎宾），跨轮次持久
        retrieval_cfg = self.config.get("retrieval", {})
        self.greeting_location_tag = str(
            retrieval_cfg.get("greeting_location_tag", GREETING_LOCATION_TAG)
        ).strip().lower() or GREETING_LOCATION_TAG
        self.greeting_location_label = str(
            retrieval_cfg.get(
                "greeting_location_label", DEFAULT_GREETING_LOCATION_LABEL
            )
        ).strip() or DEFAULT_GREETING_LOCATION_LABEL
        self._robot_location_tags: List[str] = [self.greeting_location_tag]
        self._known_tags: List[str] = self._load_known_tags()
        self._tag_pattern: Optional[re.Pattern] = self._build_tag_pattern(self._known_tags)
        logger.info("已知 tag: %s", self._known_tags)

        # 读取重排开关
        self.default_navigation_tag = str(
            retrieval_cfg.get("default_navigation_tag", "ls6")
        ).lower()
        self.use_reranker = retrieval_cfg.get("use_reranker", True)
        self.ask_name_reask_timeout_sec = float(
            retrieval_cfg.get("ask_name_reask_timeout_sec", 5000)
        )

        # 初始化 Embedding 服务（与建库同模型）
        self.embed_service = EmbeddingService(
            model_name=models.get("embedding_model", "BAAI/bge-small-zh-v1.5"),
            sentence_cache_dir=hf_runtime["sentence_cache_dir"],
            local_files_only=bool(hf_cfg.get("local_files_only", False)),
        )

        # 初始化检索器（加载 FAISS 索引）
        index_dir = Path(paths.get("index_dir", "./assets/index_store")) / "faiss_store"
        self.retriever = RetrieverRuntime(
            faiss_dir=str(index_dir),
            embeddings_model=self.embed_service,
        )

        # 初始化重排器
        if self.use_reranker:
            self.reranker = RerankerService(
                model_name=models.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
                sentence_cache_dir=hf_runtime["sentence_cache_dir"],
                local_files_only=bool(hf_cfg.get("local_files_only", False)),
            )
        else:
            self.reranker = None
            logger.info("重排已禁用（use_reranker=false），将跳过重排步骤")

        logger.info("RAGService 初始化完成（模型已加载）")

    def _load_known_tags(self) -> List[str]:
        """
        扫描 index_store/metadata/ 下的一级子目录名，作为已知 tag 集合。
        与 DP/document_loader 的 resolve_doc_tag 逻辑对应：子目录名即 tag。
        结果按字符串长度降序排列，供正则构建时保证「长 tag 优先匹配」。
        """
        paths = self.config.get("paths", {})
        metadata_dir = Path(paths.get("index_dir", "./RAG/assets/index_store")) / "metadata"
        if not metadata_dir.exists():
            return []
        tags = sorted(
            {p.name for p in metadata_dir.iterdir() if p.is_dir()},
            key=len,
            reverse=True,
        )
        return tags

    @staticmethod
    def _build_tag_pattern(tags: List[str]) -> Optional[re.Pattern]:
        """
        用已知 tag 列表构建大小写不敏感的边界正则。
        边界定义：tag 前后均不为字母或数字（兼容中文语境）。
        tags 须已按长度降序排列（长 tag 优先），避免 l6 提前匹配 ls6 的后半段。
        """
        if not tags:
            return None
        alternation = "|".join(re.escape(t) for t in tags)
        return re.compile(
            r"(?<![a-zA-Z0-9])(" + alternation + r")(?![a-zA-Z0-9])",
            re.IGNORECASE,
        )

    def _detect_tags_from_query(self, query: str) -> List[str]:
        """
        用正则从 query 中提取所有已知 tag（去重、保持 _known_tags 顺序）。
        例：「智己LS6的漆面颜色」-> ["ls6"]
        """
        if not self._tag_pattern:
            return []
        found = {m.group(1).lower() for m in self._tag_pattern.finditer(query)}
        return [t for t in self._known_tags if t in found]

    @staticmethod
    def _merge_tags(*tag_lists: List[str]) -> List[str]:
        """合并多个 tag 列表，去重并保持先出现的顺序（固定参观 tag 优先）。"""
        merged: List[str] = []
        seen: set = set()
        for tags in tag_lists:
            for tag in tags:
                normalized = (tag or "").strip().lower()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    merged.append(normalized)
        return merged

    _EXCLUDED_ROBOT_LOCATION_TAGS = frozenset({"general"})
    _EXCLUDED_ROBOT_LOCATION_PREFIXES = ("active_ask",)

    def _filter_vehicle_location_tags(self, tags: List[str]) -> List[str]:
        """从导航固定 tag 中筛出可写入机器人位置提示的展车点位（如 ls6、l6）。"""
        result: List[str] = []
        for tag in tags:
            normalized = (tag or "").strip().lower()
            if not normalized or normalized in self._EXCLUDED_ROBOT_LOCATION_TAGS:
                continue
            if any(normalized.startswith(prefix) for prefix in self._EXCLUDED_ROBOT_LOCATION_PREFIXES):
                continue
            result.append(normalized)
        return result

    def commit_extracted_name(self, vision_user_id: str, person_name: str) -> bool:
        """将姓名抽取结果写入用户状态，并重置实例级询问姓名标志。"""
        vid = (vision_user_id or "").strip()
        name = (person_name or "").strip()
        if not vid or not name:
            return False
        if not is_concrete_person_name(name):
            logger.info(
                "抽取姓名无效，跳过写入: vision_user_id=%s raw=%r", vid, person_name
            )
            return False
        self.visitor_state.set_person_name(vid, name)
        if self._pending_name_user_id == vid:
            self.second_ask = False
            self.second = False
            self._pending_name_user_id = None
        logger.info(
            "已写入抽取姓名: vision_user_id=%s person_name=%s", vid, name
        )
        return True

    def reset_robot_to_greeting_location(self) -> None:
        """将机器人当前位置重置为门口迎宾，并清除导航检索状态。"""
        self._pinned_navigation_tags = []
        self._active_tags = []
        self._robot_location_tags = [self.greeting_location_tag]
        logger.info(
            "机器人位置已重置为%s (tag=%s)",
            self.greeting_location_label,
            self.greeting_location_tag,
        )

    @staticmethod
    def starts_with_move_to_wait(llm_text: str) -> bool:
        """判断 LLM 回复最前端是否为 MOVE_TO_WAIT 意图。"""
        return bool(MOVE_TO_WAIT_INTENT_PREFIX.match(llm_text or ""))

    def handle_llm_response(self, llm_text: str) -> bool:
        """
        处理 LLM 回复文本。若最前端为 <INTENT>MOVE_TO_WAIT</INTENT>，
        则将机器人位置重置为门口迎宾位置。
        """
        if not self.starts_with_move_to_wait(llm_text):
            return False
        self.reset_robot_to_greeting_location()
        return True

    def _print_result(self, result: Dict[str, Any]) -> None:
        """打印 RAG 结果的详细信息（日志+控制台），与原有 print_rag_result 功能一致"""
        logger.info("检索耗时: %.4fs", result["timings"]["retrieve"])
        logger.info("重排耗时: %.4fs", result["timings"]["rerank"])
        logger.info(
            "上下文与提示词构建耗时: %.4fs", result["timings"]["build_context_prompt"]
        )
        logger.info("总耗时(不含LLM): %.4fs", result["timings"]["total"])

        output = {
            "query": result["query"],
            "active_tags": result.get("active_tags", []),
            "retrieved_count": len(result["retrieved_docs"]),
            "reranked_count": len(result["reranked_docs"]),
        }
        logger.info("RAG 运行时输出:\n%s", json.dumps(output, ensure_ascii=False, indent=2))

        logger.info("=" * 60)
        logger.info("context一览:")
        for idx, doc in enumerate(result["reranked_docs"], 1):
            logger.info(f"片段 {idx}:")
            logger.info(doc.get("text", ""))
            logger.info(f"元数据: {doc.get('metadata', {})}")
            logger.info(
                f"向量分: {doc.get('vector_score')}, 重排分: {doc.get('rerank_score')}"
            )
        logger.info("-" * 40)

        logger.info("=" * 60)
        logger.info("生成的 Prompt（供 LLM 使用）:")
        logger.info(result["prompt"])
        logger.info("=" * 60)

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
        vision_user_id: Optional[str] = None,
        vision_user_name: Optional[str] = None,
        voice_user_id: Optional[str] = None,
        verbose: bool = True,
        is_obtain_name: bool = False,
        is_active_ask: bool = False,
        tag: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        执行完整的 RAG 流程：检索 → 重排 → 构建上下文 → 生成 Prompt。

        is_active_ask：主动招呼模式。无顾客对话时由机器人先开口；仅检索 tag=active_ask
          的资料，并使用专用 prompt。query 可为空，空时用默认语义查询做检索。

        tag：指定检索范围的 tag 列表（对应 metadata.tag 字段，如 ["ls6", "ls7"]）。
          - 显式传入非 None 列表：直接使用并更新实例 _active_tags。
          - 传入 None（默认）：先从 query 正则解析 tag；解析到则与固定参观 tag 叠加，
            否则沿用实例 _active_tags（跨调用持久化，且始终包含固定参观 tag）。
          - 传入空列表 []：清除历史 tag 与固定参观 tag，本次全库检索。
          - is_active_ask=True 时忽略上述规则，固定检索 active_ask，且不改写 _active_tags。
          - 参观意向：命中预设地点时固定对应 tag；未命中时使用 default_navigation_tag。
            固定 tag 在本轮立即生效，直到下一次参观意向才被覆盖；期间 query 解析到的
            其他车型 tag 仅叠加，不替换固定 tag。
        """
        if is_obtain_name and is_active_ask:
            raise ValueError("is_obtain_name 与 is_active_ask 不能同时为 True")

        vid = (vision_user_id or "").strip() or None
        incoming_name = (vision_user_name or "").strip() or None
        # 兼容 CLI：仅传入人名作为 vision_user_id 时，视为人名
        if vid and not incoming_name and is_concrete_person_name(vid):
            incoming_name = vid
        logger.info(
            "query 入参 vision_user_id=%s vision_user_name=%s voice_user_id=%s is_active_ask=%s is_obtain_name=%s",
            vid,
            incoming_name,
            (voice_user_id or "").strip() or None,
            is_active_ask,
            is_obtain_name,
        )

        raw_query = (query or "").strip()
        if not raw_query and not is_active_ask:
            raise ValueError("query 不能为空")

        # 参数默认值
        retrieval_cfg = self.config.get("retrieval", {})
        vector_threshold = retrieval_cfg.get("vector_threshold", 0.6)
        # 主动招呼模式已按 tag 收窄检索范围，使用专用（更低）阈值避免全部结果被过滤
        active_ask_vector_threshold = float(
            retrieval_cfg.get("active_ask_vector_threshold", 0.0)
        )
        top_k = top_k if top_k is not None else int(retrieval_cfg.get("top_k", 8))
        rerank_top_n = (
            rerank_top_n
            if rerank_top_n is not None
            else int(retrieval_cfg.get("rerank_top_n", 4))
        )

        active_ask_tag = retrieval_cfg.get("active_ask_tag", ACTIVE_ASK_TAG)
        default_active_ask_query = retrieval_cfg.get(
            "active_ask_retrieval_query", DEFAULT_ACTIVE_ASK_RETRIEVAL_QUERY
        )

        # ── tag 解析 ─────────────────────────────────────────────────────────
        active_ask_stage_hint: str = ""
        is_navigation_turn = False
        if is_active_ask:
            # 阶段感知：若有用户 ID，根据当前导购阶段选择专属 tag 与检索词
            stage_tag = str(active_ask_tag).lower()
            stage_query = raw_query or default_active_ask_query
            if vid:
                next_step = self.visitor_state.get_next_pending_step(vid)
                if next_step:
                    step_id = next_step["id"]
                    cfg = STAGE_ACTIVE_ASK_CONFIG.get(step_id)
                    if cfg:
                        stage_tag, stage_query_tmpl, active_ask_stage_hint = cfg
                        stage_query = raw_query or stage_query_tmpl
                        logger.info(
                            "主动招呼阶段感知: step_id=%s tag=%s hint=%s",
                            step_id, stage_tag, active_ask_stage_hint,
                        )
                    else:
                        logger.info("主动招呼阶段感知: step_id=%s 无对应配置，使用通用 tag", step_id)
                else:
                    logger.info("主动招呼：用户 %s 全部阶段已完成，使用通用兜底", vid)
            resolved_tags = [stage_tag]
            retrieve_query = stage_query
            logger.info(
                "主动招呼模式: tag=%s retrieve_query=%s",
                resolved_tags,
                retrieve_query,
            )
        else:
            retrieve_query = raw_query
            visit_locations = self.config.get("visit_locations")
            detected = self._detect_tags_from_query(raw_query)

            if tag is not None:
                resolved_tags = [t.lower() for t in tag if t]
                self._active_tags = resolved_tags
                if not resolved_tags:
                    self._pinned_navigation_tags = []
                logger.info("tag 来源=显式传入: %s", resolved_tags)
            else:
                if detected:
                    resolved_tags = self._merge_tags(
                        self._pinned_navigation_tags, detected
                    )
                    self._active_tags = resolved_tags
                    logger.info(
                        "tag 来源=query 解析(含固定参观): %s", resolved_tags
                    )
                else:
                    resolved_tags = self._merge_tags(
                        self._pinned_navigation_tags, self._active_tags
                    )
                    logger.info(
                        "tag 来源=沿用历史(含固定参观): %s",
                        resolved_tags if resolved_tags else "无（全库检索）",
                    )

            if not is_obtain_name:
                nav_tags = resolve_pending_navigation_tags(
                    raw_query,
                    visit_locations,
                    self.default_navigation_tag,
                )
                if nav_tags is not None:
                    is_navigation_turn = True
                    self._pinned_navigation_tags = list(nav_tags)
                    resolved_tags = self._merge_tags(nav_tags, detected)
                    self._active_tags = resolved_tags
                    logger.info(
                        "参观意向已固定 tag: %s", self._pinned_navigation_tags
                    )
        # ─────────────────────────────────────────────────────────────────────

        # 叠加 general tag：有具体车型 tag 时同步检索通用知识库（品牌介绍、通用 FAQ 等）。
        # 仅在有具体 tag 且非主动招呼模式时叠加；全库检索（resolved_tags=[]）无需处理。
        _GENERAL_TAG = "general"
        if resolved_tags and not is_active_ask and _GENERAL_TAG not in resolved_tags:
            resolved_tags = resolved_tags + [_GENERAL_TAG]
            logger.info("叠加 general tag，最终检索 tag: %s", resolved_tags)

        user_state: Optional[Dict[str, Any]] = None
        if vid:
            user_state = self.visitor_state.get_or_create(vid)
            logger.info(
                "用户状态: vision_user_id=%s vision_user_name=%s file=%s %s ask_name=%s person_name=%s",
                vid,
                incoming_name,
                self.visitor_state.get_state_file_path(vid),
                self.visitor_state.get_tour_progress_summary(vid),
                user_state.get("ask_name"),
                user_state.get("person_name"),
            )

        # 如果调用方明确要求这是提取姓名的请求，重置实例级 second 为 False
        if is_obtain_name:
            self.second_ask = False
            self.second = False

        # user_id 切换检测：若在等待提取姓名阶段 user_id 已切换，放弃本次提取并重置，
        # 以便新顾客先走正常询问姓名流程，下一轮再提取
        if self.second_ask and vid != self._pending_name_user_id:
            logger.info(
                "user_id 已切换（%s -> %s），放弃上一轮姓名提取，将对新顾客重新询问姓名",
                self._pending_name_user_id,
                vid,
            )
            self.second_ask = False
            self.second = False

        if self.second_ask:
            self.second = True

        timings = {}

        if is_obtain_name:
            logger.info("姓名抽取模式，跳过检索与访客询问状态")
            retrieved: list = []
            reranked: list = []
            timings["retrieve"] = 0.0
            timings["rerank"] = 0.0
            context = ""
            should_ask_name = False
        else:
            # 检索（tags=None 时全库检索，tags 非空时按 tag 后过滤）
            t0 = time.perf_counter()
            retrieved = self.retriever.retrieve(
                retrieve_query,
                top_k,
                tags=resolved_tags if resolved_tags else None,
            )
            timings["retrieve"] = time.perf_counter() - t0

            # 重排（根据配置决定是否执行）
            if self.use_reranker and self.reranker is not None:
                t1 = time.perf_counter()
                reranked = self.reranker.rerank(query, retrieved, rerank_top_n)
                timings["rerank"] = time.perf_counter() - t1
            else:
                # 未启用重排时，直接使用检索结果，并截取前 rerank_top_n 条
                reranked = retrieved[:rerank_top_n]
                timings["rerank"] = 0.0

            ctx_threshold = active_ask_vector_threshold if is_active_ask else vector_threshold
            context = build_context(reranked, vector_threshold=ctx_threshold)
            should_ask_name = False

        # 构建上下文和 Prompt
        t2 = time.perf_counter()
        visit_locations = self.config.get("visit_locations")

        # 以 uuid 管理用户状态；有姓名则写入，无姓名则走询问姓名逻辑
        resolved_name: Optional[str] = None
        if not is_obtain_name and vid:
            if incoming_name and is_concrete_person_name(incoming_name):
                resolved_name = incoming_name
                self.visitor_state.set_person_name(vid, incoming_name)

            state = self.visitor_state.get(vid)
            if not resolved_name and state and state.get("person_name"):
                resolved_name = str(state["person_name"]).strip() or None

            if resolved_name:
                should_ask_name = False
            else:
                if state is None:
                    should_ask_name = True
                    self.second_ask = True
                    self._pending_name_user_id = vid
                    self.visitor_state.mark_asked(vid)
                else:
                    should_ask_name = self.visitor_state.should_reask(
                        vid, self.ask_name_reask_timeout_sec
                    )
                    if should_ask_name:
                        self.second_ask = True
                        self._pending_name_user_id = vid

        if vid and is_active_ask and not is_obtain_name:
            advanced_step = self.visitor_state.mark_next_tour_step(vid)
            if advanced_step:
                logger.info(
                    "主动招呼推进观车流程: vision_user_id=%s step_id=%s",
                    vid,
                    advanced_step,
                )
            user_state = self.visitor_state.get_or_create(vid)

        robot_location_tags: List[str] = []
        if not is_obtain_name and not is_active_ask and not is_navigation_turn:
            if self._pinned_navigation_tags:
                vehicle_tags = self._filter_vehicle_location_tags(
                    self._pinned_navigation_tags
                )
                if vehicle_tags:
                    self._robot_location_tags = vehicle_tags
                    logger.info(
                        "导航已完成，更新机器人位置: tags=%s", vehicle_tags
                    )
            robot_location_tags = list(self._robot_location_tags)

        prompt = build_prompt(
            raw_query,
            context,
            vision_user_id=vision_user_id,
            vision_user_name=resolved_name,
            voice_user_id=voice_user_id,
            visit_locations=visit_locations,
            should_ask_name=should_ask_name,
            is_obtain_name=is_obtain_name,
            is_active_ask=is_active_ask,
            active_ask_stage_hint=active_ask_stage_hint,
            robot_location_tags=robot_location_tags,
            greeting_location_tag=self.greeting_location_tag,
            greeting_location_label=self.greeting_location_label,
        )
        timings["build_context_prompt"] = time.perf_counter() - t2
        timings["total"] = (
            timings["retrieve"] + timings["rerank"] + timings["build_context_prompt"]
        )

        # ── 语言规则库：根据本轮 query 自动标记已完成的观车环节 ──────────────
        auto_steps: List[str] = []
        if vid and not is_obtain_name:
            auto_steps = detect_completed_steps(query=raw_query, response="")
            if auto_steps:
                for step_id in auto_steps:
                    self.visitor_state.mark_tour_step_asked(vid, step_id)
                user_state = self.visitor_state.get_or_create(vid)
                logger.info(
                    "语言规则自动标记: vision_user_id=%s 命中环节=[%s]",
                    vid,
                    steps_summary(auto_steps),
                )
        # ─────────────────────────────────────────────────────────────────────

        result = {
            "query": raw_query,
            "retrieve_query": retrieve_query if is_active_ask else raw_query,
            "retrieved_docs": retrieved,
            "reranked_docs": reranked,
            "context": context,
            "prompt": prompt,
            "timings": timings,
            "second": self.second,
            "active_tags": resolved_tags,
            "is_active_ask": is_active_ask,
            "vision_user_id": vid,
            "visitor_state": user_state,
            "auto_matched_steps": auto_steps,
        }
        if verbose:
            self._print_result(result)

        return result


def split_query_user_suffix(text: str) -> Tuple[str, Optional[str]]:
    """
    从输入中解析「问题 + 可选后缀  --姓名」。
    使用自右向左最后一次出现的「  --」分割，前面为检索用问题，后面为 user_id。
    """
    text = text.strip()
    if " --" not in text:
        return text, None
    question, suffix = text.rsplit(" --", 1)
    question = question.strip()
    suffix = suffix.strip()
    if not suffix:
        return text, None
    return question, suffix


def print_rag_result(result: Dict[str, Any]) -> None:
    """打印 RAG 结果的详细信息（日志+控制台）。"""
    logger.info("检索耗时: %.4fs", result["timings"]["retrieve"])
    logger.info("重排耗时: %.4fs", result["timings"]["rerank"])
    logger.info(
        "上下文与提示词构建耗时: %.4fs", result["timings"]["build_context_prompt"]
    )
    logger.info("总耗时(不含LLM): %.4fs", result["timings"]["total"])

    output = {
        "query": result["query"],
        "retrieved_count": len(result["retrieved_docs"]),
        "reranked_count": len(result["reranked_docs"]),
    }
    logger.info("RAG 运行时输出:\n%s", json.dumps(output, ensure_ascii=False, indent=2))

    logger.info("=" * 60)
    logger.info("context一览:")
    for idx, doc in enumerate(result["reranked_docs"], 1):
        logger.info(f"片段 {idx}:")
        logger.info(doc.get("text", ""))
        logger.info(f"元数据: {doc.get('metadata', {})}")
        logger.info(
            f"向量分: {doc.get('vector_score')}, 重排分: {doc.get('rerank_score')}"
        )
    logger.info("-" * 40)

    logger.info("=" * 60)
    logger.info("生成的 Prompt（供 LLM 使用）:")
    logger.info(result["prompt"])
    logger.info("=" * 60)


def interactive_mode(rag_service: RAGService, top_k: int, rerank_top_n: int) -> None:
    """交互式模式：循环读取用户输入，输出 RAG 结果，输入 exit 退出。"""
    print(
        "\n进入 RAG 交互模式，输入问题后按回车查看结果，输入 exit 或 quit 退出。\n"
        "可选在问题末尾附带姓名：「问题  --姓名」，例如：RAG是什么？ --张三"
    )
    while True:
        try:
            user_input = input("\n请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出交互模式")
            break

        if user_input.lower() in ("exit", "quit"):
            print("退出交互模式")
            break

        if not user_input:
            print("问题不能为空，请重新输入。")
            continue

        question, inline_user_id = split_query_user_suffix(user_input)
        if not question:
            print("问题不能为空（去掉「  --姓名」后无有效问题），请重新输入。")
            continue

        try:
            rag_service.query(
                question,
                top_k=top_k,
                rerank_top_n=rerank_top_n,
                vision_user_id=inline_user_id,
            )
        except Exception as e:
            print(f"处理问题时出错: {e}")
            logger.exception("交互模式查询异常")


def main() -> None:
    logger.info("启动RAG服务中...")
    args = build_parser(CONFIG).parse_args()

    # 创建服务（单次使用，脚本结束后销毁）
    rag_service = RAGService(config=CONFIG)
    logger.info("RAG服务启动完成")

    if args.query is not None:
        # 单次查询模式
        logger.info("开始 RAG 流程（检索->重排->上下文->Prompt）")
        raw = args.query.strip()
        if not raw:
            raise ValueError("query 不能为空")

        question, inline_user_id = split_query_user_suffix(raw)
        if not question:
            raise ValueError("query 不能为空（去掉「  --姓名」后无有效问题）")

        cli_user_id = (
            args.user_id.strip()
            if args.user_id is not None and str(args.user_id).strip()
            else None
        )
        user_id = cli_user_id if cli_user_id is not None else inline_user_id

        rag_service.query(
            query=question,
            top_k=args.top_k,
            rerank_top_n=args.rerank_top_n,
            vision_user_id=user_id,
        )
        logger.info("完成一次 RAG 服务流程")
    else:
        # 交互模式
        interactive_mode(rag_service, args.top_k, args.rerank_top_n)


if __name__ == "__main__":
    main()
