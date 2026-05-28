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

def _build_user_context_lines(
    vision_user_id: Optional[str],
    should_ask_name: bool,
) -> str:
    """根据传入的 vision_user_id 推断访客身份并生成提示行。

    仅使用传入的 `vision_user_id` 作为识别来源：
    - 如果该 id 包含中文字符且非时间戳，视为人名；
    - 否则视为访客标识（如时间戳或编号）。
    """
    display_id = str(vision_user_id).strip() if vision_user_id is not None and str(vision_user_id).strip() else None
    is_name = is_concrete_person_name(vision_user_id)
    source = "视觉识别" if display_id else None
    final_name = display_id if is_name and display_id else ""

    if final_name:
        logger.info("已识别具体人名 (%s): %s", source or "视觉识别", final_name)
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
            "识别到顾客初次来访，尚未获知对方真实姓名。\n"
            "本次回复必须向访客询问如何称呼，不要编造或假设姓名，"
            "也不要使用访客编号、时间戳或匿名 ID 来称呼对方。\n"
        )

    if display_id:
        logger.info("访客已询问过姓名但仍未知，不再追问: %s", display_id)
    return (
        "【访客身份】\n"
        "当前未获得对方姓名。请正常回答问题，不要使用访客编号、时间戳或匿名 ID 来称呼对方，"
        "也不要再次主动询问姓名。\n"
    )

def _build_ask_name_prompt(query: str) -> str:
    """
    构建首次询问访客姓名的专用提示词。

    与常规 RAG 提示分离，避免模型被【资料】/【问题】带偏去作答，
    并显式覆盖默认系统提示中的 <INTENT> 标签要求。
    """
    user_utterance = (query or "").strip()
    utterance_block = ""
    if user_utterance:
        utterance_block = (
            f"\n【用户刚才说】\n{user_utterance}\n"
            "（请勿回答上述内容；仅可在问候后紧接询问称呼。）\n"
        )

    template = (
        "【最高优先级 — 询问姓名】\n"
        "面前的访客初次来访，系统尚未获知对方真实姓名。\n"
        "本次回复的唯一任务是：向访客自然、礼貌地询问如何称呼。\n"
        "\n"
        "【输出要求 — 必须全部遵守】\n"
        "1. 直接输出面向访客的中文口语回复，不要输出 <INTENT>、<LOCATION> 等任何标签"
        "（本段要求覆盖默认系统提示中的状态标签规则）。\n"
        "2. 回复正文必须是询问姓名/称呼的句子，例如："
        "「您好，请问怎么称呼您？」「您好，方便告诉我您的称呼吗？」\n"
        "3. 不要回答用户问题，不要引用检索资料，不要介绍产品或展厅内容。\n"
        "4. 不要编造或假设对方姓名；不要用访客编号、时间戳或匿名 ID 称呼对方。\n"
        "5. 控制在 40 字以内，语气亲切自然。\n"
        "{utterance_block}"
        "\n"
        "请直接输出询问姓名的回复："
    )
    return template.format(utterance_block=utterance_block)


# LLM 姓名抽取常见无效输出
_EXTRACT_NAME_EMPTY_TOKENS = frozenset(
    {
        "<EMPTY>",
        '""',
        "''",
        "无",
        "未知",
        "不知道",
        "未提供",
        "空",
        "空字符串",
        "none",
        "null",
        "n/a",
    }
)

_INTENT_TAG_PATTERN = re.compile(r"<INTENT>.*?</INTENT>", re.DOTALL | re.IGNORECASE)
_LOCATION_TAG_PATTERN = re.compile(r"<LOCATION>.*?</LOCATION>", re.DOTALL | re.IGNORECASE)
_CHINESE_NAME_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,4}")


def sanitize_extracted_name(raw: str | None) -> str:
    """
    清洗 LLM 姓名抽取结果：仅保留 2–4 个连续汉字的人名，否则返回空字符串。
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    text = _INTENT_TAG_PATTERN.sub("", text)
    text = _LOCATION_TAG_PATTERN.sub("", text).strip()
    if text.lower() in _EXTRACT_NAME_EMPTY_TOKENS or text in _EXTRACT_NAME_EMPTY_TOKENS:
        return ""
    # 去掉常见包裹符号
    text = text.strip("「」『』\"' \t\n\r，。！？!?；;：:")
    if not text or text.lower() in _EXTRACT_NAME_EMPTY_TOKENS:
        return ""
    match = _CHINESE_NAME_PATTERN.search(text)
    if not match:
        return ""
    candidate = match.group(0)
    if not is_concrete_person_name(candidate):
        return ""
    return candidate


def _build_obtain_name_prompt(query: str) -> str:
    """
    构建从用户回复中抽取姓名的专用提示词。

    与常规 RAG / 询问姓名提示分离，要求模型仅输出姓名或空行，
    并显式覆盖默认系统提示中的 <INTENT> 标签要求。
    """
    user_utterance = (query or "").strip()
    utterance_block = ""
    if user_utterance:
        utterance_block = f"\n【待分析文本】\n{user_utterance}\n"

    template = (
        "【最高优先级 — 姓名抽取（非对话）】\n"
        "你是姓名抽取器。从【待分析文本】中判断访客是否在告知自己的真实中文姓名。\n"
        "本任务与展厅介绍、意图状态、RAG 资料无关。\n"
        "\n"
        "【输出契约 — 违反任意一条即错误】\n"
        "1. 只输出一行纯文本：要么是姓名本身，要么是空行（零字符，不要输出任何可见字符）。\n"
        "2. 禁止输出：<INTENT>、<LOCATION>、书名号、引号、冒号、解释、道歉、问候、"
        "「无法确定」「无」「不知道」、<EMPTY>、JSON、英文或其他任何附加内容。\n"
        "3. 姓名规则：2–4 个连续汉字，为人名用字；可去掉「我叫/叫我/是/姓」等前缀后只保留姓名。\n"
        "4. 以下情况必须输出空行（零字符）：\n"
        "   - 未提供姓名、拒绝透露、含糊其辞\n"
        "   - 仅寒暄/打招呼/呼叫机器人（如「你好」「小特」）\n"
        "   - 在问产品、展厅、价格等业务问题\n"
        "   - 文本中的汉字属于车名、地名、品牌、他人姓名，而非访客自称\n"
        "5. 不得编造姓名；不得把机器人名、品牌名、车型名当作访客姓名。\n"
        "6. 本任务禁止输出任何 INTENT 状态标签（输出标签视为失败）。\n"
        "（本段要求覆盖默认系统提示中的状态标签规则。）\n"
        "\n"
        "【示例 — 输出必须与「期望输出」完全一致】\n"
        "待分析文本：叫我张三吧\n"
        "期望输出：张三\n"
        "\n"
        "待分析文本：我姓李，名四是四行的四\n"
        "期望输出：李四\n"
        "\n"
        "待分析文本：不想告诉你我的名字\n"
        "期望输出：\n"
        "\n"
        "待分析文本：你好，小特\n"
        "期望输出：\n"
        "\n"
        "待分析文本：智己 LS6 多少钱\n"
        "期望输出：\n"
        "\n"
        "待分析文本：大家都叫我小王\n"
        "期望输出：小王\n"
        "{utterance_block}"
        "\n"
        "请只输出姓名或空行："
    )
    return template.format(utterance_block=utterance_block)


def build_prompt(
    query: str,
    context: str,
    vision_user_id: Optional[str] = None,
    voice_user_id: Optional[str] = None,
    visit_locations: Optional[List[str]] = None,
    should_ask_name: bool = False,
    is_obtain_name: bool = False,
) -> str:
    """
    组装 RAG 提示词字符串，融合访客身份、意图状态与导航地点指令。

    要求模型：
    1. 基于【资料】回答问题，资料不足则说明“资料不足”。
    2. 仅根据 vision_user_id 与状态参数调整称呼策略（voice_user_id 暂不处理）。
    3. 根据用户问题判断意图状态，并在回答末尾附加 <INTENT>状态</INTENT>。
    4. 参观意向且含具体地点时，在 <INTENT> 后附加 <LOCATION>地点</LOCATION>。
    """
    if is_obtain_name:
        logger.info("构建姓名抽取专用 prompt")
        return _build_obtain_name_prompt(query)

    # 首次见面需询问姓名时，使用专用 prompt，避免与 RAG 作答指令及默认 INTENT 规则冲突。
    if should_ask_name:
        logger.info("构建询问姓名专用 prompt")
        return _build_ask_name_prompt(query)

    user_line = _build_user_context_lines(
        vision_user_id=vision_user_id,
        should_ask_name=should_ask_name,
    )
    intent_line, detected_location = _build_intent_instruction_lines(query, visit_locations)
    if detected_location:
        logger.info("参观意向已识别地点: %s", detected_location)

    base_instruction = (
        "基于【资料】回答问题。如果资料不足以回答问题，请明确回复“抱歉，我无法回答你的问题”。不要自己编造答案。优先以【资料】中的内容回答问题，如果【资料】中的内容不足以回答问题，则根据历史对话记录合理推测每次问题的主语，并结合【资料】中的内容回答问题。\n"
    )

    closing = (
        "请基于【资料】输出完整回答。充分理解并应用【资料】中的内容，使得回答丰富且有深度。"
        + "并按照【意图状态】要求输出 <INTENT> 标签"
        + ("及 <LOCATION> 标签（如适用）" if detected_location else "")
        + "："
    )

    template = (
        "你是一个智能导购助手。请严格遵循以下要求：\n"
        "{base_instruction}\n"
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
        base_instruction=base_instruction,
        user_line=user_line,
        intent_line=intent_line,
        context=context,
        question=query,
        closing=closing,
    )
