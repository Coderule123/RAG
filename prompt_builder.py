import re
from typing import List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from RAG.config.logger_runtime import get_logger

logger = get_logger("rag")

# vision_user_id 为时间戳时的格式：年_月_日_分_秒
TIMESTAMP_USER_ID_PATTERN = re.compile(r"^\d{4}_\d{1,2}_\d{1,2}_\d{1,2}_\d{1,2}$")

# 可导航地点（示例：A、B、C；可按展厅实际名称扩展）
DEFAULT_VISIT_LOCATIONS: List[str] = ["A", "B", "C"]

# 参观/游览意向关键词
VISIT_INTENT_KEYWORDS: Tuple[str, ...] = (
    "参观",
    "去看看",
    "想看",
    "想去",
    "带我去",
    "带我看看",
    "逛逛",
    "看一下",
    "看一看",
    "游览",
    "逛逛展厅",
)

def is_visit_intent_query(query: str) -> bool:
    """判断用户问题是否体现参观/游览意向。"""
    q = (query or "").strip()
    if not q:
        return False
    return any(kw in q for kw in VISIT_INTENT_KEYWORDS)


def extract_visit_location(query: str,locations: Optional[List[str]] = None,) -> Optional[str]:
    """
    从问题中提取可导航地点。
    仅当问题包含参观意向且命中已知地点列表时返回地点（多个则逗号拼接）。
    """
    if not is_visit_intent_query(query):
        return None
    locs = locations if locations is not None else DEFAULT_VISIT_LOCATIONS
    matched = [loc for loc in locs if loc and loc in query]
    if not matched:
        return None
    return ",".join(matched)

def _build_intent_instruction_lines(query: str,locations: Optional[List[str]] = None,) -> Tuple[str, Optional[str]]:
    """
    生成意图与地点相关的输出格式说明。
    返回 (instruction_text, detected_location)。
    """
    detected_location = extract_visit_location(query, locations)

    intent_rules = ""

    if detected_location:
        intent_rules += (
            f"【参观意向】用户问题已提到地点：{detected_location}。\n"
            f"如果状态为 <INTENT>NEEDS_GUIDANCE</INTENT> 则之后紧跟 <LOCATION>{detected_location}</LOCATION>，"
            "并在正文中简要说明将引导对方前往该地点。\n"
        )
    elif is_visit_intent_query(query):
        intent_rules += (
            "【参观意向】\n用户问题体现参观/游览意向，但未指明具体地点（"
            + "、".join(locations or DEFAULT_VISIT_LOCATIONS)
            + "）。可主动询问想参观哪个区域，暂不输出 <LOCATION> 标签。\n"
        )

    return intent_rules, detected_location

def is_timestamp_user_id(user_id: str) -> bool:
    """判断是否为匿名访客时间戳标识（年_月_日_分_秒）。"""
    return bool(TIMESTAMP_USER_ID_PATTERN.match(user_id.strip()))


def is_concrete_person_name(user_id: Optional[str]) -> bool:
    """
    判断 user_id 是否为具体人名。
    时间戳格式视为非人名；含中文且非时间戳则视为人名（）。
    """
    if user_id is None:
        return False
    raw = str(user_id).strip()
    if not raw:
        return False
    if is_timestamp_user_id(raw):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", raw))


def resolve_display_name(vision_user_id: Optional[str] = None,) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    解析用于提示词展示的用户标识。
    返回 (标识字符串, 是否为具体人名, 识别来源说明)。
    """
    candidates: List[Tuple[str, str]] = []
    if vision_user_id is not None and str(vision_user_id).strip():
        candidates.append(("视觉识别", str(vision_user_id).strip()))

    for source, uid in candidates:
        if is_concrete_person_name(uid):
            return uid, True, source

    if candidates:
        source, uid = candidates[0]
        return uid, False, source
    return None, False, None

def _build_user_context_lines(
    vision_user_id: Optional[str],
    should_ask_name: bool,
    resolved_name: Optional[str],
) -> str:
    """根据姓名状态生成访客身份相关提示行。"""
    display_id, is_name, source = resolve_display_name(vision_user_id)
    final_name = (resolved_name or "").strip()

    if not final_name and is_name and display_id:
        final_name = display_id

    if final_name:
        logger.info("已识别具体人名 (%s): %s", source or "状态提取", final_name)
        return (
            "【访客身份】\n"
            f"当前已知用户姓名是：{final_name}。请在回答中适时、礼貌地使用对方姓名（如「{final_name}」），语气亲切自然；"
            "不必每句话都重复姓名，避免生硬。\n"
        )

    if should_ask_name:
        if display_id:
            logger.info("访客标识首次询问姓名: %s", display_id)
        return (
            "【访客身份】\n"
            "当前仅识别到访客标识，尚未获知对方真实姓名。请在回答中自然、礼貌地询问面前的人如何称呼；"
            "不要编造或假设姓名，也不要使用访客编号、时间戳或匿名 ID 来称呼对方。\n"
        )

    if display_id:
        logger.info("访客已询问过姓名但仍未知，不再追问: %s", display_id)
    return (
        "【访客身份】\n"
        "当前未获得对方姓名。请正常回答问题，不要使用访客编号、时间戳或匿名 ID 来称呼对方，"
        "也不要再次主动询问姓名。\n"
    )

def build_prompt(
    query: str,
    context: str,
    vision_user_id: Optional[str] = None,
    voice_user_id: Optional[str] = None,
    visit_locations: Optional[List[str]] = None,
    should_ask_name: bool = False,
    resolved_name: Optional[str] = None,
) -> str:
    """
    组装 RAG 提示词字符串，融合访客身份、意图状态与导航地点指令。

    要求模型：
    1. 基于【资料】回答问题，资料不足则说明“资料不足”。
    2. 仅根据 vision_user_id 与状态参数调整称呼策略（voice_user_id 暂不处理）。
    3. 根据用户问题判断意图状态，并在回答末尾附加 <INTENT>状态</INTENT>。
    4. 参观意向且含具体地点时，在 <INTENT> 后附加 <LOCATION>地点</LOCATION>。
    """
    user_line = _build_user_context_lines(
        vision_user_id=vision_user_id,
        should_ask_name=should_ask_name,
        resolved_name=resolved_name,
    )
    intent_line, detected_location = _build_intent_instruction_lines(query, visit_locations)
    if detected_location:
        logger.info("参观意向已识别地点: %s", detected_location)

    _, is_name, _ = resolve_display_name(vision_user_id)
    has_name = bool((resolved_name or "").strip()) or is_name
    closing = (
        "请基于【资料】输出完整回答。"
        + ("回答中请礼貌、恰当地使用对方姓名。" if has_name else "")
        + ("并在回答中自然询问对方姓名。" if should_ask_name else "")
        + "并按照【意图状态】要求输出 <INTENT> 标签"
        + ("及 <LOCATION> 标签（如适用）" if detected_location else "")
        + "："
    )

    template = (
        "你是一个智能导购助手。请严格遵循以下要求：\n"
        "基于【资料】回答问题。如果资料不足以回答问题，请明确回复“抱歉，我无法回答你的问题”。\n"
        "\n"
        "{user_line}\n"
        "{intent_line}\n"
        "【资料】\n"
        "{context}\n"
        "\n"
        "【问题】\n"
        "{question}\n"
        "\n"
        "{closing}"
    )
    prompt = ChatPromptTemplate.from_template(template)
    return prompt.format(
        user_line=user_line,
        intent_line=intent_line,
        context=context,
        question=query,
        closing=closing,
    )
