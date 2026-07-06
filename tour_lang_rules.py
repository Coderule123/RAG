"""
观车讲解流程 — 语言规则库

用途
----
对每轮对话的【用户问题】和【机器人回复】进行文本匹配，
自动判断哪些销售环节已经讲解/完成，并返回对应的 step_id 列表。

规则设计原则
------------
1. 每个规则包含 any_of（满足任意一条即命中）和 all_of（需同时满足全部才命中，可为空）。
2. 对话文本取 query（用户输入）+ response（机器人回复）的合并串进行匹配。
3. 规则尽量短、高精度，避免误判；宁可漏，不要错。
4. 同一环节可被多条规则命中（幂等），不会重复标记。
"""

import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Set


class StepRule(NamedTuple):
    step_id: str
    any_of: List[str]   # 正则列表：命中任意一条即触发
    all_of: List[str]   # 正则列表：需全部命中才触发（与 any_of 结果取交集）
    description: str    # 规则说明（仅用于调试）


# ──────────────────────────────────────────────────────────────
# 规则库（按 step_id 对应 DEFAULT_TOUR_STEPS 的 8 个阶段）
# ──────────────────────────────────────────────────────────────
TOUR_STEP_RULES: List[StepRule] = [

    # ① 接待问候 —— 问候语、询问是否首次来访、自我介绍主理人
    StepRule(
        step_id="greeting",
        any_of=[
            r"欢迎.*智己",
            r"第一次.*来.*店",
            r"来.*店.*吗",
            r"您好.*智己",
            r"智己.*欢迎",
            r"用户主理人",
            r"请问.*怎么称呼",
            r"怎么称呼.*您",
        ],
        all_of=[],
        description="进店问候 / 首次来访判断 / 自我介绍",
    ),

    # ② 探寻需求 —— 用车场景、谁开、纯电/增程倾向、关注车型
    StepRule(
        step_id="needs_exploration",
        any_of=[
            r"家人.*开",
            r"自己.*开",
            r"用车.*场景",
            r"关注.*车型",
            r"纯电.*增程",
            r"增程.*纯电",
            r"购车.*意向",
            r"看.*什么车",
            r"想看.*哪",
            r"需求.*了解",
            r"用车.*主要",
        ],
        all_of=[],
        description="探寻用车人、用车场景与车型偏好",
    ),

    # ③ 动力续航 —— 增程技术、800V、续航里程、用车成本
    StepRule(
        step_id="powertrain_range",
        any_of=[
            r"超级增程",
            r"骁遥",
            r"1500.*公里",
            r"续航.*公里",
            r"800V",
            r"充电.*分钟",
            r"亏电.*油耗",
            r"每公里.*分钱",
            r"充一次电",
            r"纯电.*续航.*45",
            r"增程.*技术",
        ],
        all_of=[],
        description="超级增程/纯电续航/800V补能/使用成本",
    ),

    # ④ 车外讲解 —— 底盘、后轮转向、安全、智驾激光雷达
    StepRule(
        step_id="exterior_chassis",
        any_of=[
            r"灵蜥.*底盘",
            r"前双叉臂",
            r"后轮转向",
            r"18.*度.*转角",
            r"转弯半径",
            r"防侧翻",
            r"零自燃",
            r"爆胎.*稳定",
            r"激光雷达",
            r"Momenta",
            r"momenta",
            r"智驾.*全程",
            r"超级后驱",
        ],
        all_of=[],
        description="车外：底盘/后轮转向/安全/超级后驱/智驾",
    ),

    # ⑤ 主驾体验 —— 大屏、雨夜模式、一键泊车/代驾
    StepRule(
        step_id="driver_cockpit",
        any_of=[
            r"27.*英寸",
            r"5K.*屏",
            r"Mini.*LED",
            r"主驾.*屏",
            r"雨夜模式",
            r"补盲显示",
            r"一键泊车",
            r"一键贴边",
            r"一键循迹",
            r"一键脱困",
            r"坐.*主驾",
            r"上车.*体验",
            r"AI.*代驾",
        ],
        all_of=[],
        description="主驾座舱：5K大屏/雨夜模式/一键AI代驾",
    ),

    # ⑥ 副驾后排 —— 零重力、贵妃椅、后排空间、大冰箱、后备箱
    StepRule(
        step_id="copilot_rear",
        any_of=[
            r"零重力",
            r"贵妃椅",
            r"头等舱.*布局",
            r"副驾.*屏",
            r"3K.*屏",
            r"后排.*空间",
            r"双开门.*冰箱",
            r"14罐",
            r"后备箱",
            r"行李箱",
        ],
        all_of=[],
        description="副驾/后排：零重力/贵妃椅/冰箱/后备箱",
    ),

    # ⑦ 邀请试驾 —— 明确提出试驾邀请
    StepRule(
        step_id="test_drive",
        any_of=[
            r"试驾",
            r"开一开",
            r"体验.*驾驶",
            r"亲自.*开",
            r"上路.*感受",
            r"安排.*试驾",
        ],
        all_of=[],
        description="邀请/确认试驾",
    ),

    # ⑧ 购买意向 —— 价格、金融、订单、优惠
    StepRule(
        step_id="purchase_intent",
        any_of=[
            r"多少钱",
            r"价格",
            r"优惠",
            r"金融.*方案",
            r"贷款",
            r"月供",
            r"下订",
            r"订车",
            r"交付",
            r"交定金",
            r"几折",
        ],
        all_of=[],
        description="价格/金融/下订意向确认",
    ),
]

# 预编译所有正则，提升重复调用性能
_COMPILED_RULES: List[Dict] = []
for _rule in TOUR_STEP_RULES:
    _COMPILED_RULES.append({
        "step_id": _rule.step_id,
        "any_of": [re.compile(p, re.IGNORECASE) for p in _rule.any_of],
        "all_of": [re.compile(p, re.IGNORECASE) for p in _rule.all_of],
        "description": _rule.description,
    })


def detect_completed_steps(
    query: str,
    response: str = "",
    extra_context: str = "",
) -> List[str]:
    """
    对本轮对话文本（query + response + extra_context）运行全部规则，
    返回命中的 step_id 列表（保持规则定义顺序，去重）。

    参数
    ----
    query        : 用户输入文本
    response     : 机器人回复文本（可为空，仅用 query 也能匹配）
    extra_context: 附加文本（如 RAG 上下文片段关键词），通常留空
    """
    text = " ".join(filter(None, [query, response, extra_context]))
    matched: List[str] = []
    seen: Set[str] = set()

    for rule in _COMPILED_RULES:
        if rule["step_id"] in seen:
            continue

        hit_any = any(pat.search(text) for pat in rule["any_of"])
        hit_all = all(pat.search(text) for pat in rule["all_of"]) if rule["all_of"] else True

        if hit_any and hit_all:
            matched.append(rule["step_id"])
            seen.add(rule["step_id"])

    return matched


def detect_steps_from_query_only(query: str) -> List[str]:
    """仅对用户问题做匹配（不含机器人回复），用于提前预判阶段。"""
    return detect_completed_steps(query=query)


def get_rule_descriptions() -> Dict[str, str]:
    """返回每个 step_id 对应的规则说明，供调试使用。"""
    return {rule["step_id"]: rule["description"] for rule in _COMPILED_RULES}


def steps_summary(step_ids: Sequence[str]) -> str:
    """将 step_id 列表格式化为可读字符串，用于日志输出。"""
    descs = get_rule_descriptions()
    return "、".join(descs.get(sid, sid) for sid in step_ids) if step_ids else "无"
