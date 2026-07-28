import re
from typing import List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from RAG.config.logger_runtime import get_logger

logger = get_logger("rag")

# 主动招呼模式：仅检索 data/active_ask/ 下文档（metadata.tag=active_ask）
ACTIVE_ASK_TAG = "active_ask"
# 无用户发言时用于向量检索的默认语义查询
DEFAULT_ACTIVE_ASK_RETRIEVAL_QUERY = (
    "展厅主动招呼 欢迎语 引导顾客交流 开口话术 接待用语"
)

# vision_user_id 为时间戳时的格式：年_月_日_分_秒
TIMESTAMP_USER_ID_PATTERN = re.compile(r"^\d{4}_\d{1,2}_\d{1,2}_\d{1,2}_\d{1,2}$")

# 机器人退下意图：LLM 回复最前端命中时重置为门口迎宾位置
MOVE_TO_WAIT_INTENT_PREFIX = re.compile(
    r"^\s*<INTENT>MOVE_TO_WAIT</INTENT>",
    re.IGNORECASE,
)
GREETING_LOCATION_TAG = "greeting"
DEFAULT_GREETING_LOCATION_LABEL = "门口迎宾位置"

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
            f"【参观意向】用户问题已命中预设地点：{detected_location}。\n"
            f"若意图状态为 NEEDS_GUIDANCE，标签必须严格输出为 <INTENT>NEEDS_GUIDANCE<LOCATION>{detected_location}</LOCATION></INTENT>，"
            f"<LOCATION> 内只能填写 {detected_location}，禁止填写任何其他文字。"
            "并在正文中简要说明将引导对方前往该地点。\n"
        )
    elif is_visit_intent_query(query):
        loc_list = "、".join(locations or DEFAULT_VISIT_LOCATIONS)
        intent_rules += (
            "【参观意向】\n"
            "用户表达了参观/游览意向，但问题中未命中任何预设地点（"
            + loc_list
            + "）。\n"
            "本次必须询问用户想参观哪个区域；"
            "严格禁止输出 <LOCATION> 标签，也禁止将用户话语中的任意文字作为地点值写入标签。\n"
        )

    return intent_rules, detected_location


def resolve_pending_navigation_tags(
    query: str,
    locations: Optional[List[str]] = None,
    default_tag: str = "ls6",
) -> Optional[List[str]]:
    """
    参观意向时计算应固定到检索范围的 tag 列表；非参观意向返回 None。

    - 命中预设地点：tag 为命中的地点（多个地点则多个 tag）
    - 未命中地点：tag 为 default_tag（导航默认展区，如 ls6）

    由 RAGService 在本轮立即写入 _pinned_navigation_tags，直到下次参观意向覆盖。
    """
    if not is_visit_intent_query(query):
        return None
    detected = extract_visit_location(query, locations)
    if detected:
        return [t.strip().lower() for t in detected.split(",") if t.strip()]
    default = (default_tag or "ls6").strip().lower()
    return [default] if default else None


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
    vision_user_name: Optional[str] = None,
) -> str:
    """根据 vision_user_id（uuid）与 vision_user_name 生成访客身份提示行。"""
    display_id = (
        str(vision_user_id).strip()
        if vision_user_id is not None and str(vision_user_id).strip()
        else None
    )
    known_name = None
    if vision_user_name and str(vision_user_name).strip():
        name = str(vision_user_name).strip()
        if is_concrete_person_name(name):
            known_name = name

    if known_name:
        logger.info("已识别具体人名 (视觉识别): %s", known_name)
        return (
            "【访客身份】\n"
            f"当前已知用户姓名是：{known_name}。请在回答中适时、礼貌地使用对方姓名（如「{known_name}」），语气亲切自然；"
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


# 展车 tag 与对外车型名（与 data/l6、data/ls6 等目录一致）
VEHICLE_TAG_DISPLAY_NAMES: dict[str, str] = {
    "l6": "智己L6",
    "ls6": "智己LS6",
    "ls7": "智己LS7",
    "ls8": "智己LS8",
    "ls9": "智己LS9",
}
_EXCLUDED_VEHICLE_REFERENCE_TAGS = frozenset({"general"})
_EXCLUDED_VEHICLE_REFERENCE_PREFIXES = ("active_ask",)


def _filter_vehicle_tags(
    tags: Optional[List[str]],
    greeting_location_tag: str = GREETING_LOCATION_TAG,
) -> List[str]:
    """从 tag 列表中筛出展车点位（排除 general、greeting、active_ask* 等）。"""
    greeting = (greeting_location_tag or "").strip().lower()
    result: List[str] = []
    for tag in tags or []:
        normalized = (tag or "").strip().lower()
        if not normalized or normalized == greeting:
            continue
        if normalized in _EXCLUDED_VEHICLE_REFERENCE_TAGS:
            continue
        if any(normalized.startswith(prefix) for prefix in _EXCLUDED_VEHICLE_REFERENCE_PREFIXES):
            continue
        result.append(normalized)
    return result


def _resolve_single_vehicle_tag(
    active_tags: Optional[List[str]] = None,
    robot_location_tags: Optional[List[str]] = None,
    greeting_location_tag: str = GREETING_LOCATION_TAG,
) -> Optional[str]:
    """检索 tag 与机器人位置合并后，若仅锁定单一展车则返回其 tag。"""
    vehicle_tags = _filter_vehicle_tags(active_tags, greeting_location_tag)
    for tag in _filter_vehicle_tags(robot_location_tags, greeting_location_tag):
        if tag not in vehicle_tags:
            vehicle_tags.append(tag)
    unique = list(dict.fromkeys(vehicle_tags))
    return unique[0] if len(unique) == 1 else None


def _build_factual_accuracy_lines() -> str:
    """引导 LLM 严守资料边界，禁止凭营销话术或常识推断配置事实。"""
    return (
        "【事实准确性 — 全车型配置】\n"
        "回答任何车型的价格、续航、加速、座位数、座椅加热/通风/按摩/4D、零重力、"
        "空悬、冰箱、后排屏、智驾硬件等配置问题时，必须优先采信【资料】中含"
        "「官方指导价」「参数表」「CLTC」「标配/选配/无」等结构化字段的内容。\n"
        "当【资料】中营销话术写「全系」「标配」「全排」「前排+后排」等，"
        "与同条或他处参数表字段（如「选配」「无」、未分列座位）冲突时，"
        "必须以参数表为准，不得采信营销话术中的配置分布。\n"
        "用户询问「哪些座椅/哪几个位置/标配还是选装」时，"
        "必须仅在【资料】有明确座位、位置或参数表字段时作答；"
        "禁止根据场景描述（如「妈妈久坐二排」）、功能泛述、同排类推或行业常识推断。\n"
        "禁止把某一座椅的功能推广到同排其他座椅或其他排次；"
        "禁止将5座车型说成有三排，或将「3个零重力座椅」说成「第三排零重力」。\n"
        "若【资料】只介绍功能特点但未写明配备位置或版本，"
        "只回答功能本身，并说明「具体以当前门店配置表为准」。\n"
    )


def _build_robot_capability_boundary_lines() -> str:
    """引导 LLM 区分机器人可执行能力与不可执行的物理/线下动作承诺。"""
    return (
        "【机器人能力边界】\n"
        "你是展厅智能助手小特，只能通过语音讲解和引导参观展区（触发导航），不能执行任何物理动作。\n"
        "你可以做的：介绍品牌/车型/配置/参数，解答疑问，引导用户梳理需求，"
        "在用户明确表达参观意向时引导前往展区（配合 <INTENT>NEEDS_GUIDANCE>）。\n"
        "禁止承诺或暗示自己能：递水/拿东西、开关车门、上车演示、操作车机按钮、取车钥匙、"
        "亲自安排或陪同试驾、签合同、办理保险/上牌/验车、查询库存排产、替用户联系第三方等线下事务。\n"
        "当【资料】出现上述不可执行的动作承诺时，必须改写，不得照搬：\n"
        "- 讲解类：改为「我可以为您介绍……」「我来给您讲讲……」\n"
        "- 体验/试驾类：改为「如需试驾/进一步体验，可以联系我们的用户主理人安排」\n"
        "- 手续/交付类：改为「用户主理会协助您办理……」「我可以先为您介绍流程」\n"
        "- 参观类：仅在用户有参观意向时说「咱们可以过去那边看看」\n"
        "禁止输出「我来帮你操作」「我带您去取钥匙」「我给您拿瓶水」等你无法兑现的承诺。\n"
    )


def _build_natural_oral_style_lines() -> str:
    """引导 LLM 将资料改写为自然导购口语，避免照抄检索原文。"""
    return (
        "【输出风格 — 自然口语】\n"
        "参考【资料】中的事实信息与导购意图作答，但必须改写为面向顾客的自然口语，"
        "禁止照搬原文、禁止罗列条目、禁止像念稿。"
        "语气应像站在展车旁的真实导购，亲切自然、简洁有条理。\n"
    )


def _build_conversation_history_usage_lines() -> str:
    """
    引导 LLM 使用当前会话中已恢复的 HumanMessage/AIMessage 历史。

    历史对话由 LLM Agent 按人脸/视觉 ID（thread_id）从 db checkpointer 自动恢复，
    无需在 RAG prompt 中重复拼接历史文本。
    """
    return (
        "【对话历史 — 使用方式】\n"
        "当前会话上下文中已包含基于人脸 ID 恢复的历史 HumanMessage/AIMessage，"
        "请直接阅读这些消息作为对这位用户的记忆。\n"
        "用户已经问过、已经确认或已经得到回答的信息，不要再次追问；"
        "应自然承接（例如「刚才您关注的……」），表现出记得对方此前说过什么。\n"
        "当用户本轮使用「这个」「刚才那个」「它」「还有呢」等省略表达，"
        "或未明确说明车型、版本、配置、预算、用途等主语时，"
        "优先结合上述历史消息补全语义，再结合【资料】作答；"
        "不要把历史来源说成数据库、记录或系统信息。\n"
    )


def _build_vehicle_reference_lines(
    active_tags: Optional[List[str]] = None,
    robot_location_tags: Optional[List[str]] = None,
    greeting_location_tag: str = GREETING_LOCATION_TAG,
) -> str:
    """单一车型上下文时，引导 LLM 用「咱们这款车」等口语指代，避免生硬重复车型名。"""
    single_tag = _resolve_single_vehicle_tag(
        active_tags, robot_location_tags, greeting_location_tag
    )
    if not single_tag:
        return ""
    display_name = VEHICLE_TAG_DISPLAY_NAMES.get(single_tag, single_tag.upper())
    logger.info("单一车型上下文: tag=%s display=%s", single_tag, display_name)
    return (
        "【口径 — 车型指代】\n"
        f"当前对话已锁定单一车型（{display_name}），顾客正在看或咨询这款车。\n"
        "回答时用「咱们这款车」「我们这款车」「这款车」等展厅导购口语指代，"
        f"不要每句都以「{display_name}」开头或重复堆砌车型名；语气应像站在展车旁的真实导购。\n"
        "但以下情况仍须保留完整车型/版本名称：用户明确对比多个车型；"
        "回答涉及具体款型/配置版本（如 Max、Ultra、74kWh）或精确参数数值时；"
        "【资料】原文中的版本名、价格档位可直接引用，无需改成「咱们这款车」。\n"
    )


def _build_robot_location_lines(
    robot_location_tags: Optional[List[str]],
    greeting_location_tag: str = GREETING_LOCATION_TAG,
    greeting_location_label: str = DEFAULT_GREETING_LOCATION_LABEL,
) -> str:
    """导航完成后或处于迎宾区时，将机器人当前位置写入 prompt。"""
    tags = [t.strip() for t in (robot_location_tags or []) if t and str(t).strip()]
    if not tags:
        return ""
    if len(tags) == 1 and tags[0] == greeting_location_tag:
        logger.info("机器人位置提示: %s", greeting_location_label)
        return (
            "【机器人位置】\n"
            f"机器人当前在{greeting_location_label}。"
            f"回答时可自然说明当前在门店入口/迎宾区，例如「我现在在{greeting_location_label}」。\n"
        )
    if len(tags) == 1:
        display = tags[0]
        example = f"我现在旁边的车是{display}"
    else:
        display = "、".join(tags)
        example = f"我现在旁边的车是{tags[0]}"
    logger.info("机器人位置提示: %s", display)
    return (
        "【机器人位置】\n"
        f"机器人已完成导航，当前所在展车旁，旁边的车是 {display}。"
        f"回答时可自然引用当前位置，例如「{example}」。\n"
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


# LLM 姓名抽取常见无效输出（道歉、问候、否定及 LLM 拒答用语）
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
        "抱歉",
        "对不起",
        "您好",
        "你好",
        "无法",
        "不能",
        "无法确定",
        "不确定",
    }
)

# LLM 拒答/解释句中常见片段（出现即视为非姓名输出）
_EXTRACT_NAME_REFUSAL_KEYWORDS = frozenset(
    {
        "抱歉",
        "对不起",
        "无法",
        "不能",
        "未能",
        "未识别",
        "无法识别",
        "无法从",
        "不能从",
        "无法确定",
        "不确定",
        "不知道",
        "未提供",
        "无有效",
        "无效",
        "识别出",
        "提取",
        "抽取",
    }
)

_INTENT_TAG_PATTERN = re.compile(r"<INTENT>.*?</INTENT>", re.DOTALL | re.IGNORECASE)
_LOCATION_TAG_PATTERN = re.compile(r"<LOCATION>.*?</LOCATION>", re.DOTALL | re.IGNORECASE)
_SENTENCE_PUNCT_PATTERN = re.compile(r"[，。！？；：,\.!?;:]")
_VALID_NAME_PATTERN = re.compile(r"^[\u4e00-\u9fff]{2,4}$")
_NAME_INTRO_PREFIX_PATTERN = re.compile(
    r"^(?:大家都?(?:都)?叫我|叫我|我?(?:叫|是|姓)|名字(?:叫|是)?|称呼(?:是|叫)?)\s*"
)
_NAME_SUFFIX_PARTICLES = "吧呢啊呀哦哈嘛"

# 「姓氏+称谓」输出形式（如「刘先生」「王女士」），默认称谓为「先生」
_HONORIFIC_SUFFIXES = ("先生", "女士", "小姐")
_DEFAULT_HONORIFIC = "先生"

# 常见复姓（「我姓欧阳」→「欧阳先生」）
_COMPOUND_SURNAMES = frozenset(
    {
        "欧阳", "司马", "上官", "诸葛", "东方", "皇甫", "尉迟", "公孙",
        "令狐", "慕容", "长孙", "宇文", "司徒", "司空", "夏侯", "独孤",
        "南宫", "西门", "东郭", "百里", "呼延", "澹台", "端木", "申屠",
    }
)

# 常见单字姓氏（现代常用姓氏 + 百家姓高频段），用于校验 LLM 只输出单个姓氏时的兜底转换
_COMMON_SINGLE_SURNAMES = frozenset(
    # 现代人口排名靠前的常用姓氏
    "王李张刘陈杨黄赵吴周徐孙马朱胡郭何林罗高郑梁谢宋唐许韩冯邓曹彭曾"
    "肖田董潘袁蔡蒋余于杜叶程魏苏吕丁任卢姚沈钟姜崔谭陆范汪廖石金韦贾"
    "夏付方邹熊白孟秦邱侯江尹薛闫段雷龙黎史陶贺毛郝顾龚邵万覃武钱戴严"
    "莫孔向汤常温康施牛樊葛邢路岳齐易伍乔贲庞倪"
    # 百家姓高频段补充
    "褚卫尤华陶戚喻柏水窦章云奚郎鲁昌苗凤花俞柳酆鲍费廉岑滕殷毕邬安乐"
    "时傅皮卞元卜平和穆湛祁禹狄米贝明臧计伏成谈茅纪舒屈项祝阮蓝闵席季"
    "麻娄危童颜梅盛刁骆凌霍虞支柯昝管经房裘缪干解应宗宣郁单杭洪包诸左"
    "吉钮"
)

# 姓名主体中不应出现的虚词/代词/否定词/常见动词（防止「我叫不出来」误抽为「不出来」）
_NAME_STOPCHARS = set(
    "的了吗呢吧啊呀哦哈嘛不没啥谁这那它他她你我您和跟与什么怎么去来到看走"
)

# 规则抽取：高置信「报全名」引导词（刻意排除歧义大的「我是」，交给 LLM 兜底）
# 允许捕获 1 字：如「我叫刘」只留下姓氏时按「姓氏+先生」处理
_FULLNAME_INTRO_PATTERN = re.compile(
    r"(?:我叫|叫我|我名叫|名字(?:叫|是))\s*([\u4e00-\u9fff]{1,4})"
)

# 规则抽取：「只报姓氏」引导词（我姓刘 / 免贵姓王 / 敝姓张）
_SURNAME_INTRO_PATTERN = re.compile(
    r"(?:我姓|免贵姓|敝姓|本人姓|鄙人姓)\s*([\u4e00-\u9fff]{1,2})"
)


def _is_plausible_name_chars(text: str) -> bool:
    """姓名主体不应包含虚词、代词、否定词等停止字符。"""
    return bool(text) and not any(ch in _NAME_STOPCHARS for ch in text)


def extract_name_by_rules(text: Optional[str]) -> str:
    """
    规则版姓名抽取：LLM 调用前的确定性快速路径。

    仅处理高置信模式，命中即返回，未命中返回空字符串交由 LLM 抽取：
    - 「我叫张三 / 叫我张三 / 名字叫张三」 → 张三
    - 「我姓刘 / 免贵姓王」 → 刘先生（复姓如「我姓欧阳」→ 欧阳先生）
    """
    if not text:
        return ""
    utterance = str(text).strip()
    if not utterance:
        return ""

    # 1) 报全名模式
    match = _FULLNAME_INTRO_PATTERN.search(utterance)
    if match:
        candidate = match.group(1).rstrip(_NAME_SUFFIX_PARTICLES)
        if _is_plausible_name_chars(candidate):
            if len(candidate) == 1:
                # 「我叫刘」等只留下单字时按姓氏处理
                return f"{candidate}{_DEFAULT_HONORIFIC}"
            if 2 <= len(candidate) <= 4:
                return candidate

    # 2) 只报姓氏模式 → 姓氏+先生
    match = _SURNAME_INTRO_PATTERN.search(utterance)
    if match:
        candidate = match.group(1).rstrip(_NAME_SUFFIX_PARTICLES)
        if candidate in _COMPOUND_SURNAMES:
            return f"{candidate}{_DEFAULT_HONORIFIC}"
        surname = candidate[:1]
        if surname and _is_plausible_name_chars(surname):
            return f"{surname}{_DEFAULT_HONORIFIC}"

    return ""


def _normalize_honorific_name(name: str) -> Optional[str]:
    """
    处理「姓氏+称谓」形式（刘先生/王女士/欧阳先生）。

    返回 None 表示不含称谓后缀（交由普通姓名校验）；
    返回空字符串表示含称谓但主体无效；否则返回规范化后的「主体+称谓」。
    """
    for suffix in _HONORIFIC_SUFFIXES:
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if (
                re.fullmatch(r"[\u4e00-\u9fff]{1,4}", base)
                and _is_plausible_name_chars(base)
            ):
                return base + suffix
            return ""
    return None


def _strip_name_affixes(text: str) -> str:
    """去掉「我叫/叫我/姓」等前缀及语气词后缀，保留疑似姓名片段。"""
    name = text.strip("「」『』\"' \t\n\r")
    for _ in range(3):
        stripped = _NAME_INTRO_PREFIX_PATTERN.sub("", name, count=1).strip()
        if stripped == name:
            break
        name = stripped
    return name.rstrip(_NAME_SUFFIX_PARTICLES).strip()


def _looks_like_name_refusal(text: str) -> bool:
    """判断是否为 LLM 拒答/解释性长句，而非纯姓名输出。"""
    if _SENTENCE_PUNCT_PATTERN.search(text):
        return True
    if any(keyword in text for keyword in _EXTRACT_NAME_REFUSAL_KEYWORDS):
        return True
    # 全名最长 4 字，「姓氏+称谓」形式最长 6 字（如「欧阳先生」「叫我刘先生」剥前缀前）
    if len(re.findall(r"[\u4e00-\u9fff]", text)) > 6:
        return True
    return False


def sanitize_extracted_name(raw: str | None) -> str:
    """
    清洗 LLM 姓名抽取结果：
    - 2–4 个连续汉字人名（可剥离「我叫/叫我/姓」等前后缀）原样返回；
    - 「姓氏+先生/女士/小姐」形式（如「刘先生」「欧阳先生」）规范化后返回；
    - 只输出单个常见姓氏时（如「刘」）自动转换为「刘先生」；
    - 拒答句、解释句、问候语等均返回空字符串。
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

    if _looks_like_name_refusal(text):
        return ""

    name = _strip_name_affixes(text)
    if not name or name.lower() in _EXTRACT_NAME_EMPTY_TOKENS or name in _EXTRACT_NAME_EMPTY_TOKENS:
        return ""

    # 「姓氏+称谓」形式：如「刘先生」「王女士」，主体有效则保留称谓返回
    honorific_name = _normalize_honorific_name(name)
    if honorific_name is not None:
        return honorific_name

    # 只输出单个姓氏时（如「刘」），按「姓氏+先生」兜底（仅限常见姓氏，防止误判）
    if len(name) == 1:
        if name in _COMMON_SINGLE_SURNAMES:
            return f"{name}{_DEFAULT_HONORIFIC}"
        return ""

    if not _VALID_NAME_PATTERN.fullmatch(name):
        return ""
    if not is_concrete_person_name(name):
        return ""
    return name


def _build_active_ask_prompt(query: str, context: str, stage_hint: str = "") -> str:
    """
    构建「主动询问顾客」专用提示词。

    场景：顾客无有效对话时由机器人先开口，或机器人主动推进导购流程。
    stage_hint 由 RAGService 根据用户当前导购阶段注入，告知模型本轮应推进到哪个环节。
    """
    hint = (query or "").strip()
    hint_block = ""
    if hint:
        hint_block = (
            f"\n【可选情境提示】\n{hint}\n"
            "（仅供你把握语气与侧重点，勿当作顾客已提出的问题，也不要逐字复述。）\n"
        )

    stage_block = ""
    if stage_hint:
        stage_block = (
            f"\n【当前导购阶段目标】\n{stage_hint}\n"
            "请你围绕上述阶段目标，生成符合该阶段的主动引导语，"
            "自然地推动对话向下一步进展，不要跳跃到尚未触及的阶段。\n"
        )

    context_block = (context or "").strip()
    if not context_block:
        context_block = "（暂无检索到对应话术资料，请结合阶段目标用简短自然的口语开场。）"

    capability_boundary_line = _build_robot_capability_boundary_lines()
    natural_oral_style_line = _build_natural_oral_style_lines()

    template = (
        "【最高优先级 — 主动发起对话（导购推进模式）】\n"
        "当前没有收到顾客的有效提问，由你主动开口。\n"
        "你的角色是展厅智能导购助手小特，像一位真人专业导购一样，"
        "自然、礼貌地引导顾客进入下一个体验环节。\n"
        "当前会话上下文中已包含基于人脸 ID 恢复的历史 HumanMessage/AIMessage；"
        "必须先阅读这些历史消息，判断顾客已经表达过的需求、疑虑、车型偏好、预算和体验反馈；"
        "不要重复询问已经回答过的问题，应在【当前导购阶段目标】内选择最适合继续推进的一句主动询问或引导。\n"
        "\n"
        "{capability_boundary_line}\n"
        "{natural_oral_style_line}\n"
        "【输出要求 — 必须全部遵守】\n"
        "1. 直接输出面向顾客的一句或两句中文口语，禁止输出 <INTENT>、<LOCATION> 等任何标签。\n"
        "2. 必须结合【当前导购阶段目标】，让话语有明确的引导方向，而不是泛泛问候。\n"
        "3. 必须结合会话中的历史 HumanMessage/AIMessage 选择最合适的下一问：缺什么问什么，已确认的内容只简短承接，不重复盘问。\n"
        "4. 可参考【主动话术资料】中的风格与范例，但必须遵守【输出风格 — 自然口语】与【机器人能力边界】改写后输出。\n"
        "5. 禁止在回复中出现资料分类名、文件夹名、tag 名（如 active_ask、ls6 等）"
        "及「根据资料」「检索」等系统用语。\n"
        "6. 若已知顾客姓名，可适当使用（如「张先生」），语气亲切自然；"
        "未知姓名则不要臆造称呼，也不要重复询问。\n"
        "7. 不要编造未经核实的价格、配置或政策数据。\n"
        "8. 控制在 80 字以内，亲切自然，不压迫，不过度推销。\n"
        "{stage_block}"
        "{hint_block}"
        "\n"
        "【主动话术资料】\n"
        "{context}\n"
        "\n"
        "请直接输出你对顾客说的引导语："
    )
    return template.format(
        capability_boundary_line=capability_boundary_line,
        natural_oral_style_line=natural_oral_style_line,
        stage_block=stage_block,
        hint_block=hint_block,
        context=context_block,
    )


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
        "1. 只输出一行纯文本：要么是姓名（或「姓氏+先生」），要么是空行（零字符，不要输出任何可见字符）。\n"
        "2. 禁止输出：<INTENT>、<LOCATION>、书名号、引号、冒号、解释、道歉、问候、"
        "「无法确定」「无」「不知道」、<EMPTY>、JSON、英文或其他任何附加内容。\n"
        "3. 姓名规则：2–4 个连续汉字，为人名用字；可去掉「我叫/叫我/是/姓」等前缀后只保留姓名。\n"
        "   特别地：若访客只告知姓氏而未报全名（如「我姓刘」「免贵姓王」），"
        "必须输出「姓氏+先生」，例如「刘先生」「王先生」；复姓同理（「我姓欧阳」→「欧阳先生」）。\n"
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
        "待分析文本：我姓刘\n"
        "期望输出：刘先生\n"
        "\n"
        "待分析文本：免贵姓王\n"
        "期望输出：王先生\n"
        "\n"
        "待分析文本：我姓欧阳\n"
        "期望输出：欧阳先生\n"
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


def build_echo_name_prompt(person_name: str) -> str:
    """
    规则快速路径命中姓名后的「原样回显」专用提示词。

    姓名已由 extract_name_by_rules 确定（并写入用户状态），
    LLM 仅需原样输出该姓名，以保持「抽取调用返回姓名」的接口契约不变。
    """
    name = (person_name or "").strip()
    return (
        "【最高优先级 — 固定输出（非对话）】\n"
        "本次任务只有一个要求：只输出下面「目标内容」本身，"
        "不要添加任何其他字符、标点、引号、标签或解释。\n"
        "禁止输出 <INTENT>、<LOCATION> 等任何状态标签"
        "（本段要求覆盖默认系统提示中的状态标签规则）。\n"
        "\n"
        f"【目标内容】\n{name}\n"
        "\n"
        f"请只输出：{name}"
    )


def build_prompt(
    query: str,
    context: str,
    vision_user_id: Optional[str] = None,
    voice_user_id: Optional[str] = None,
    vision_user_name: Optional[str] = None,
    visit_locations: Optional[List[str]] = None,
    should_ask_name: bool = False,
    is_obtain_name: bool = False,
    is_active_ask: bool = False,
    active_ask_stage_hint: str = "",
    robot_location_tags: Optional[List[str]] = None,
    active_tags: Optional[List[str]] = None,
    greeting_location_tag: str = GREETING_LOCATION_TAG,
    greeting_location_label: str = DEFAULT_GREETING_LOCATION_LABEL,
) -> str:
    """
    组装 RAG 提示词字符串，融合访客身份、意图状态与导航地点指令。

    要求模型：
    1. 基于【资料】回答问题，资料不足则说明“资料不足”。
    2. 根据 vision_user_id（uuid）与 vision_user_name 调整称呼策略（voice_user_id 暂不处理）。
    3. 根据用户问题判断意图状态，并在回答末尾附加 <INTENT>状态</INTENT>。
    4. 参观意向且含具体地点时，在 <INTENT> 后附加 <LOCATION>地点</LOCATION>。
    """
    if is_obtain_name:
        logger.info("构建姓名抽取专用 prompt")
        return _build_obtain_name_prompt(query)

    if is_active_ask:
        logger.info("构建主动招呼专用 prompt stage_hint=%s", active_ask_stage_hint or "(无)")
        return _build_active_ask_prompt(query, context, stage_hint=active_ask_stage_hint)

    # 首次见面需询问姓名时，使用专用 prompt，避免与 RAG 作答指令及默认 INTENT 规则冲突。
    if should_ask_name:
        logger.info("构建询问姓名专用 prompt")
        return _build_ask_name_prompt(query)

    user_line = _build_user_context_lines(
        vision_user_id=vision_user_id,
        should_ask_name=should_ask_name,
        vision_user_name=vision_user_name,
    )
    robot_location_line = _build_robot_location_lines(
        robot_location_tags,
        greeting_location_tag=greeting_location_tag,
        greeting_location_label=greeting_location_label,
    )
    vehicle_reference_line = _build_vehicle_reference_lines(
        active_tags=active_tags,
        robot_location_tags=robot_location_tags,
        greeting_location_tag=greeting_location_tag,
    )
    intent_line, detected_location = _build_intent_instruction_lines(query, visit_locations)
    if detected_location:
        logger.info("参观意向已识别地点: %s", detected_location)

    history_usage_line = _build_conversation_history_usage_lines()
    capability_boundary_line = _build_robot_capability_boundary_lines()
    factual_accuracy_line = _build_factual_accuracy_lines()
    natural_oral_style_line = _build_natural_oral_style_lines()

    base_instruction = (
        "基于【资料】回答问题，不要编造未经资料支持的具体价格、参数、配置或政策。\n"
        "回答车型配置、功能和参数问题时，优先使用【资料】里的明确字段；如果用户问法较口语，"
        "要把参数表字段转换成自然导购口径，例如把「近光灯/远光灯/自适应远近光/灯光特色功能」"
        "整合回答为「车灯有哪些功能」。\n"
        "如果【资料】没有精确数值或某一版本的细项，不要直接输出「抱歉」「无法回答」「资料不足」；"
        "应先说明已知的相关信息，再用「具体以当前门店配置表为准」等方式弱化未知细节。"
        "只有当问题完全脱离车辆、品牌、导购和展厅范围，且会话历史也无法补足主语时，才简短说明暂时无法确认。\n"
        "如果【资料】中的内容不足以完整回答问题，可结合会话历史推测问题主语（如车型、功能名称），"
        "但不得据此编造【资料】未写明的配置分布、数量、参数或政策；"
        "对未写明部分必须用「具体以当前门店配置表为准」等方式处理，不得凭猜测补全。\n"
    )

    if detected_location:
        location_reminder = (
            f"及 <LOCATION> 标签（<LOCATION> 内只能填写预设值 {detected_location}，禁止填写其他文字）"
        )
    else:
        location_reminder = "（本次禁止输出 <LOCATION> 标签）" if is_visit_intent_query(query) else ""

    closing = (
        "请基于【资料】输出完整回答。吸收【资料】中的事实信息与导购意图，"
        "按【机器人能力边界】【事实准确性 — 全车型配置】和【输出风格 — 自然口语】改写后作答，使得回答丰富且有深度。"
        + f"并按照【意图状态】要求输出 <INTENT> 标签{location_reminder}："
    )

    template = (
        "你是一个智能导购助手，你的名字叫小特。请严格遵循以下要求：\n"
        "{base_instruction}\n"
        "\n"
        "{capability_boundary_line}\n"
        "{factual_accuracy_line}\n"
        "{natural_oral_style_line}\n"
        "{history_usage_line}\n"
        "{user_line}\n"
        "{robot_location_line}\n"
        "{vehicle_reference_line}\n"
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
        capability_boundary_line=capability_boundary_line,
        factual_accuracy_line=factual_accuracy_line,
        natural_oral_style_line=natural_oral_style_line,
        history_usage_line=history_usage_line,
        user_line=user_line,
        robot_location_line=robot_location_line,
        vehicle_reference_line=vehicle_reference_line,
        intent_line=intent_line,
        context=context,
        question=query,
        closing=closing,
    )
