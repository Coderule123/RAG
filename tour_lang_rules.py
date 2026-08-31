"""
观车讲解流程 — 语言规则库

用途
----
对每轮对话的【用户问题】和【机器人回复】进行文本匹配，
自动判断哪些销售环节已经触达/完成，并返回对应的 step_id 列表。

规则设计原则
------------
1. 每个规则包含 any_of（满足任意一条即命中）和 all_of（需同时满足全部才命中，可为空）。
2. 对话文本取 query（用户输入）+ response（机器人回复）的合并串进行匹配。
3. 优先高精度，宁可漏报也不误报：
   - 避免使用单个高频通用词（如"外观""续航""价格"单独出现）
   - 多用复合短语，确保语境清晰后再触发
4. 同一环节可被多条规则命中（幂等），不会重复标记。
5. 阶段顺序：greeting → interest_probe → needs_analysis → vehicle_selection
             → product_presentation → test_drive → quote_negotiation
             → deal_confirmation → contact_retention
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

    # ① 展厅破冰 —— 问候语、询问是否首次来访、自我介绍
    StepRule(
        step_id="greeting",
        any_of=[
            r"欢迎.*智己",
            r"智己.*欢迎",
            r"您好.*智己",
            r"第一次.*来.*店",
            r"来.*店.*吗",
            r"用户主理人",
            r"请问.*怎么称呼",
            r"怎么称呼.*您",
            r"我是.*导购|我是.*AI|我是.*小特",
        ],
        all_of=[],
        description="进店问候 / 首次来访判断 / 自我介绍",
    ),

    # ② 意向摸底 —— 了解来意、粗粒度兴趣方向、购车成熟度
    # 与 needs_analysis 的区别：这里是高层次的"为什么来""大概看什么"，
    # 不涉及具体预算/场景等深度问题
    StepRule(
        step_id="interest_probe",
        any_of=[
            r"随便.*看看",
            r"(已经|比较).*有.*意向",
            r"打算.*买",
            r"有.*换车.*计划",
            r"来.*了解.*一下",
            r"今天.*主要.*想",
            r"(感兴趣|关注).*哪.*车",
            r"想看.*哪.*车",
            r"看.*什么.*车",
            r"对.*车.*感兴趣",
            r"来.*看.*轿车|来.*看.*SUV",
            r"纯电.*还是.*增程|增程.*还是.*纯电",
            r"(首次|第一次).*(购车|买车)",
            r"(购车|买车).*意向",
            r"只是.*逛逛",
        ],
        all_of=[],
        description="意向摸底：来意/兴趣方向/购车成熟度",
    ),

    # ③ 深度需求 —— 用途场景、预算区间、决策人、换购原因
    # 比 interest_probe 更深入，需要明确数字或具体场景描述
    StepRule(
        step_id="needs_analysis",
        any_of=[
            r"家人.*开",
            r"自己.*开",
            r"用车.*场景",
            r"主要.*用途",
            r"平时.*用车",
            r"用车.*主要",
            r"预算.*多少",
            r"预算.*区间",
            r"多少.*预算",
            r"谁.*决策",
            r"谁.*开",
            r"换车.*原因",
            r"旧车.*置换",
            r"城市.*通勤",
            r"长途.*出行",
            r"接送.*家人",
            r"后排.*乘客",
            r"几口人",
            r"最关心.*(续航|空间|安全|智能|舒适|价格)",
        ],
        all_of=[],
        description="深度需求：用途/预算/决策人/换购原因",
    ),

    # ④ 车型推荐 —— 根据需求匹配车型、版本和配置方向
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
            r"纯电.*增程.*怎么选",
            r"增程.*纯电.*怎么选",
            r"选.*纯电.*还是.*增程",
            r"(LS9|LS8|LS7|LS6|L6|L7).*(怎么样|适合|推荐|对比)",
        ],
        all_of=[],
        description="车型推荐：车型/版本/配置匹配",
    ),

    # ⑤ 车辆展示 —— 六方位绕车介绍、核心卖点讲解
    # 关键收紧：不再用单个通用词（外观/续航/空间等）触发，
    # 需要"介绍/讲解/展示 + 具体维度"的组合，或专属展示词汇
    StepRule(
        step_id="product_presentation",
        any_of=[
            r"六方位",
            r"绕车",
            r"(带您|帮您|我来|给您).*(介绍|讲|看|展示)",
            r"(介绍|讲解|展示|看看).*(外观|内饰|座舱|空间|底盘|配置|动力)",
            r"(外观|内饰|座舱|空间|底盘).*(介绍|讲解|展示|看看|怎么样)",
            r"核心.*卖点|主要.*亮点",
            r"零重力.*座椅|贵妃椅",
            r"灵蜥.*底盘",
            r"前双叉臂|后轮转向",
            r"激光雷达.*(位置|几个|哪里)",
            r"(智驾|智能驾驶|辅助驾驶).*(怎么|介绍|讲|体验)",
        ],
        all_of=[],
        description="车辆展示：需有介绍/讲解语境或专属展示词",
    ),

    # ⑥ 试乘试驾 —— 明确提出试驾邀请、路线说明、试后反馈
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

    # ⑦ 报价协商 —— 价格、权益、金融、置换与异议处理
    # 关键收紧：移除单独的"贵""便宜"等歧义词，改为复合语境短语
    StepRule(
        step_id="quote_negotiation",
        any_of=[
            r"多少钱",
            r"(整体|落地|裸车).*价",
            r"落地价|裸车价",
            r"报价",
            r"有.*优惠",
            r"权益.*怎么",
            r"金融.*方案",
            r"(贷款|按揭).*(怎么|多少|方案)",
            r"月供.*多少",
            r"首付.*多少",
            r"利率.*多少",
            r"置换.*价格",
            r"补贴.*多少",
            r"几折",
            r"预算.*不够",
            r"(觉得|感觉).*(贵|贵了)",
            r"能.*优惠.*多少",
            r"最低.*多少",
        ],
        all_of=[],
        description="报价协商：价格/权益/金融/置换（需复合语境）",
    ),

    # ⑧ 成交确认 —— 配置颜色、库存、下订、合同、定金
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
            r"(有|查).*现车",
            r"现车.*库存",
            r"颜色.*选|选.*颜色",
            r"配置.*确认|确认.*配置",
        ],
        all_of=[],
        description="成交确认：配置/库存/下订",
    ),

    # ⑨ 留档跟进 —— 留联系方式、预约回访、邀请关注
    # 替代原 delivery_explanation + after_sales_followup：
    # 这两个阶段在展厅场景中机器人几乎不参与，
    # 改为"留联系方式/预约再访"这个展厅机器人最高价值的收尾动作
    StepRule(
        step_id="contact_retention",
        any_of=[
            r"加.*微信",
            r"留.*电话",
            r"留.*联系方式",
            r"联系方式.*留",
            r"预约.*回访",
            r"预约.*到店",
            r"预约.*再来",
            r"再来.*看看",
            r"下次.*来",
            r"感谢.*到访",
            r"欢迎.*再次.*来",
            r"关注.*小程序",
            r"扫.*二维码",
            r"有问题.*联系.*我",
        ],
        all_of=[],
        description="留档跟进：留联系方式/预约回访/邀请关注",
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
