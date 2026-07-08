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
# 规则库（按 step_id 对应 DEFAULT_TOUR_STEPS 的 9 个阶段）
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

    # ② 需求分析 —— 用途、预算、决策人、换购原因、关注点
    StepRule(
        step_id="needs_analysis",
        any_of=[
            r"家人.*开",
            r"自己.*开",
            r"用车.*场景",
            r"主要.*用途",
            r"平时.*用车",
            r"关注.*车型",
            r"购车.*意向",
            r"看.*什么车",
            r"想看.*哪",
            r"需求.*了解",
            r"用车.*主要",
            r"预算.*多少",
            r"预算.*区间",
            r"谁.*决策",
            r"谁.*开",
            r"换车.*原因",
            r"旧车.*置换",
            r"最关心.*(续航|空间|安全|智能|舒适|价格)",
        ],
        all_of=[],
        description="需求分析：用途/预算/决策人/关注点",
    ),

    # ③ 车型推荐 —— 根据需求匹配车型、版本和配置方向
    StepRule(
        step_id="vehicle_selection",
        any_of=[
            r"推荐.*车型",
            r"适合.*车型",
            r"适合.*版本",
            r"哪款.*适合",
            r"哪一款.*适合",
            r"版本.*怎么选",
            r"配置.*怎么选",
            r"帮.*选.*车",
            r"根据.*需求.*推荐",
            r"家用.*推荐",
            r"通勤.*推荐",
            r"长途.*推荐",
            r"纯电.*增程.*选择",
            r"增程.*纯电.*选择",
            r"选.*纯电.*还是.*增程",
        ],
        all_of=[],
        description="车型推荐：车型/版本/配置匹配",
    ),

    # ④ 车辆展示 —— 六方位绕车、外观、底盘、座舱、空间、安全、智能
    StepRule(
        step_id="product_presentation",
        any_of=[
            r"六方位",
            r"绕车",
            r"外观",
            r"内饰",
            r"座舱",
            r"空间",
            r"后排",
            r"后备箱",
            r"动力",
            r"续航",
            r"补能",
            r"充电",
            r"安全",
            r"智驾",
            r"智能驾驶",
            r"辅助驾驶",
            r"底盘",
            r"悬架",
            r"屏幕",
            r"座椅",
            r"卖点",
            r"亮点",
            r"介绍.*配置",
            r"灵蜥.*底盘",
            r"前双叉臂",
            r"后轮转向",
            r"激光雷达",
            r"零重力",
            r"贵妃椅",
        ],
        all_of=[],
        description="车辆展示：六方位/配置/卖点体验",
    ),

    # ⑤ 试乘试驾 —— 明确提出试驾邀请、路线说明、试后反馈
    StepRule(
        step_id="test_drive",
        any_of=[
            r"试驾",
            r"试乘",
            r"开一开",
            r"体验.*驾驶",
            r"亲自.*开",
            r"上路.*感受",
            r"安排.*试驾",
            r"试驾.*路线",
            r"试驾.*预约",
            r"开下来.*感觉",
            r"驾驶.*感受",
        ],
        all_of=[],
        description="试乘试驾：邀约/路线/反馈",
    ),

    # ⑥ 报价协商 —— 价格、权益、金融、置换与异议处理
    StepRule(
        step_id="quote_negotiation",
        any_of=[
            r"多少钱",
            r"价格",
            r"报价",
            r"优惠",
            r"权益",
            r"金融.*方案",
            r"贷款",
            r"月供",
            r"首付",
            r"利率",
            r"置换",
            r"补贴",
            r"保险",
            r"落地价",
            r"裸车价",
            r"几折",
            r"贵",
            r"便宜",
            r"预算.*不够",
        ],
        all_of=[],
        description="报价协商：价格/权益/金融/置换",
    ),

    # ⑦ 成交确认 —— 配置颜色、库存、下订、合同、定金
    StepRule(
        step_id="deal_confirmation",
        any_of=[
            r"下订",
            r"订车",
            r"锁单",
            r"交定金",
            r"定金",
            r"合同",
            r"签约",
            r"成交",
            r"今天.*定",
            r"现在.*定",
            r"库存",
            r"现车",
            r"颜色.*选",
            r"配置.*确认",
        ],
        all_of=[],
        description="成交确认：配置/库存/下订",
    ),

    # ⑧ 交车说明 —— 交付周期、验车、功能讲解、售后对接
    StepRule(
        step_id="delivery_explanation",
        any_of=[
            r"交车",
            r"交付",
            r"提车",
            r"交付.*周期",
            r"多久.*提",
            r"验车",
            r"上牌",
            r"保险.*办理",
            r"交车.*流程",
            r"功能.*讲解",
            r"售后.*对接",
            r"保养.*说明",
        ],
        all_of=[],
        description="交车说明：交付/验车/用车事项",
    ),

    # ⑨ 售后跟进 —— 回访、保养、服务提醒、转介绍
    StepRule(
        step_id="after_sales_followup",
        any_of=[
            r"售后",
            r"回访",
            r"保养",
            r"质保",
            r"维修",
            r"服务群",
            r"用车.*提醒",
            r"首保",
            r"保养.*周期",
            r"客户.*关怀",
            r"转介绍",
        ],
        all_of=[],
        description="售后跟进：回访/保养/服务关怀",
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
