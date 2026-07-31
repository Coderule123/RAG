"""
用户喜好感知 — 车型与关注点规则库

用途
----
从用户问题（及可选机器人回复）中识别：
1. 询问过的车型（如 ls6 / ls7）
2. 对该车型关注的信息点（底盘、外饰、内饰、价格等）

规则尽量短、高精度；同一车型/关注点可重复命中（由状态层做幂等累计）。
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple


DEFAULT_VEHICLE_TAGS: Tuple[str, ...] = ("l6", "ls6", "ls7", "ls8", "ls9")

# 车型别名：中文/大小写写法 -> 标准 tag
VEHICLE_ALIASES: Dict[str, str] = {
    "l6": "l6",
    "智己l6": "l6",
    "ls6": "ls6",
    "智己ls6": "ls6",
    "ls7": "ls7",
    "智己ls7": "ls7",
    "ls8": "ls8",
    "智己ls8": "ls8",
    "ls9": "ls9",
    "智己ls9": "ls9",
}


class TopicRule(NamedTuple):
    topic_id: str
    title: str
    any_of: List[str]
    description: str


# 关注点规则：可按业务继续扩展
TOPIC_RULES: List[TopicRule] = [
    TopicRule(
        topic_id="chassis",
        title="底盘",
        any_of=[r"底盘", r"悬架", r"减震", r"后轮转向", r"转弯半径", r"灵蜥"],
        description="底盘/悬架/转向",
    ),
    TopicRule(
        topic_id="exterior",
        title="外饰",
        any_of=[r"外饰", r"外观", r"车漆", r"车身颜色", r"轮毂", r"大灯", r"灯组"],
        description="外观外饰",
    ),
    TopicRule(
        topic_id="interior",
        title="内饰",
        any_of=[r"内饰", r"座舱", r"中控", r"屏幕", r"座椅", r"零重力", r"贵妃椅", r"氛围灯"],
        description="内饰座舱",
    ),
    TopicRule(
        topic_id="trunk",
        title="后备箱",
        any_of=[r"后备箱", r"行李箱", r"尾箱", r"储物空间", r"装载"],
        description="后备箱/装载",
    ),
    TopicRule(
        topic_id="color",
        title="颜色",
        any_of=[r"颜色", r"配色", r"车色", r"内饰色", r"什么色"],
        description="外观/内饰颜色",
    ),
    TopicRule(
        topic_id="adas",
        title="智驾",
        any_of=[
            r"智驾",
            r"智能驾驶",
            r"辅助驾驶",
            r"自动驾驶",
            r"激光雷达",
            r"NOA",
            r"高快",
            r"城市领航",
            r"泊车",
        ],
        description="智能驾驶",
    ),
    TopicRule(
        topic_id="handling",
        title="操控",
        any_of=[r"操控", r"驾驶感", r"加速", r"动力", r"刹车", r"转向手感", r"开起来"],
        description="操控/驾驶感受",
    ),
    TopicRule(
        topic_id="price",
        title="价格",
        any_of=[
            r"多少钱",
            r"价格",
            r"报价",
            r"优惠",
            r"权益",
            r"落地价",
            r"裸车价",
            r"预算",
            r"月供",
            r"金融",
            r"贷款",
        ],
        description="价格/金融",
    ),
    TopicRule(
        topic_id="range",
        title="续航",
        any_of=[r"续航", r"里程", r"补能", r"充电", r"增程", r"纯电续航", r"亏电油耗"],
        description="续航补能",
    ),
    TopicRule(
        topic_id="space",
        title="空间",
        any_of=[r"空间", r"后排", r"腿部空间", r"乘坐", r"几个人坐", r"家用空间"],
        description="乘坐空间",
    ),
    TopicRule(
        topic_id="safety",
        title="安全",
        any_of=[r"安全", r"气囊", r"碰撞", r"电池安全", r"防侧翻"],
        description="安全配置",
    ),
    TopicRule(
        topic_id="comfort",
        title="舒适",
        any_of=[r"舒适", r"隔音", r"静音", r"冰箱", r"空调", r"通风", r"加热", r"按摩"],
        description="舒适配置",
    ),
    TopicRule(
        topic_id="config",
        title="配置",
        any_of=[r"配置", r"版本", r"哪款", r"顶配", r"标配", r"选装"],
        description="版本配置",
    ),
    TopicRule(
        topic_id="test_drive",
        title="试驾",
        any_of=[r"试驾", r"试乘", r"开一开", r"预约试驾"],
        description="试乘试驾",
    ),
    TopicRule(
        topic_id="delivery",
        title="交付",
        any_of=[r"交付", r"交车", r"提车", r"多久到", r"排产", r"现车", r"库存"],
        description="交付库存",
    ),
]


_COMPILED_TOPICS: List[Dict] = [
    {
        "topic_id": rule.topic_id,
        "title": rule.title,
        "patterns": [re.compile(p, re.IGNORECASE) for p in rule.any_of],
        "description": rule.description,
    }
    for rule in TOPIC_RULES
]

_TOPIC_TITLE_MAP: Dict[str, str] = {rule.topic_id: rule.title for rule in TOPIC_RULES}


def _normalize_vehicle_tag(raw: str) -> Optional[str]:
    key = (raw or "").strip().lower().replace(" ", "")
    if not key:
        return None
    return VEHICLE_ALIASES.get(key)


def build_vehicle_pattern(vehicle_tags: Optional[Sequence[str]] = None) -> re.Pattern:
    """构建车型匹配正则：长 tag 优先，避免 l6 抢先匹配 ls6。"""
    tags = [t.strip().lower() for t in (vehicle_tags or DEFAULT_VEHICLE_TAGS) if t]
    aliases = sorted(
        {alias for alias, tag in VEHICLE_ALIASES.items() if tag in tags},
        key=len,
        reverse=True,
    )
    if not aliases:
        aliases = list(DEFAULT_VEHICLE_TAGS)
    alternation = "|".join(re.escape(a) for a in aliases)
    return re.compile(
        r"(?<![a-zA-Z0-9])(" + alternation + r")(?![a-zA-Z0-9])",
        re.IGNORECASE,
    )


def detect_vehicles(
    text: str,
    vehicle_tags: Optional[Sequence[str]] = None,
) -> List[str]:
    """从文本中识别车型 tag，去重并保持首次出现顺序。"""
    if not text:
        return []
    pattern = build_vehicle_pattern(vehicle_tags)
    found: List[str] = []
    seen: Set[str] = set()
    for match in pattern.finditer(text):
        tag = _normalize_vehicle_tag(match.group(1))
        if tag and tag not in seen:
            seen.add(tag)
            found.append(tag)
    return found


def detect_topics(text: str) -> List[str]:
    """从文本中识别关注点 topic_id，保持规则定义顺序。"""
    if not text:
        return []
    matched: List[str] = []
    for rule in _COMPILED_TOPICS:
        if any(pat.search(text) for pat in rule["patterns"]):
            matched.append(rule["topic_id"])
    return matched


def detect_preferences(
    query: str = "",
    response: str = "",
    vehicle_tags: Optional[Sequence[str]] = None,
    fallback_vehicles: Optional[Sequence[str]] = None,
) -> Dict[str, List[str]]:
    """
    识别本轮对话中的车型与关注点。

    返回:
        {
          "vehicles": ["ls6", ...],
          "topics": ["price", "chassis", ...],
        }

    若命中关注点但未命中车型，则回退使用 fallback_vehicles（如上一轮关注车型）。
    """
    text = " ".join(filter(None, [query, response]))
    vehicles = detect_vehicles(text, vehicle_tags=vehicle_tags)
    topics = detect_topics(query)  # 关注点以用户问题为主，避免被导购话术带偏
    if not vehicles and topics:
        fallback = [
            _normalize_vehicle_tag(v) or str(v).strip().lower()
            for v in (fallback_vehicles or [])
            if v
        ]
        vehicles = [v for v in fallback if v]
    return {"vehicles": vehicles, "topics": topics}


def topic_title(topic_id: str) -> str:
    return _TOPIC_TITLE_MAP.get(topic_id, topic_id)


def preferences_summary(vehicles: Sequence[str], topics: Sequence[str]) -> str:
    topic_names = [topic_title(t) for t in topics]
    vehicle_part = "、".join(vehicles) if vehicles else "无车型"
    topic_part = "、".join(topic_names) if topic_names else "无关注点"
    return f"车型=[{vehicle_part}] 关注点=[{topic_part}]"
