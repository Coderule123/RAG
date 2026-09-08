"""
顾客原话记忆（泛用，不绑定具体卖点或家庭情况）。

为什么需要
----------
导购阶段（tour_process.asked）只知道「需求环节走过了」，不知道顾客答了什么。
LangGraph 的 max_tokens 只裁剪发给模型的会话消息，也不会把原话写成稳定档案。
主动询问时若只靠 RAG 话术范例，任何已经回答过的问题都可能被换皮再问一遍。

本模块不维护「自己开 / 单身 / 通勤」这类封闭标签。
只做三件与具体内容无关的事：
1. 判断一段文本能不能当作顾客原话收下（排除中控指令、空话）。
2. 用字面覆盖度判断检索到的话术是不是在重复「已经问过 / 顾客已经说过」的内容。
3. 把顾客原话拼进向量检索词，让 FAISS 偏向「已确认 …、不要再问同一句」。

只认顾客 query，不认机器人回复。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

# 中控主动询问指令前缀，与 visitor_state._ACTIVE_ASK_QUERY_PREFIX 保持一致。
_ACTIVE_ASK_PREFIX = "请求主动询问"

# 纯应和、无信息量，不值得写入档案。
_BACKCHANNEL = re.compile(r"^(嗯+|啊+|哦+|呃+|好的?|是的?|对)$")

# 按句号/问号切开，抽出更像问句的片段；长寒暄不会稀释覆盖度。
_CLAUSE_SPLIT = re.compile(r"[。！？!?；;]+")

# 中文疑问线索。用来从长句里切「真正在问的那截」，不解析问的是谁开还是预算。
_QUESTION_HINTS = ("还是", "吗", "呢", "哪", "什么", "多少", "几")

# 问句对问句：较短一方的 bigram 被覆盖到该比例，视为换皮同问。
# 0.32 能拦住「自己开还是家里一起用」的长短换皮，又不会误杀完全不同的二选一问句。
RETRIEVAL_QUESTION_COVERAGE = 0.32

# 顾客原话对检索话术：原话更短，公共词（「预算大概」）容易抬高覆盖度，所以阈值更高。
# 0.50：能拦住话术里复述了原话的主要片段，避免「说过预算」误杀「全款还是分期」。
RETRIEVAL_UTTERANCE_COVERAGE = 0.50


def is_system_active_ask_query(query: str) -> bool:
    """中控「请求主动询问…」不是顾客原话。"""
    return (query or "").strip().startswith(_ACTIVE_ASK_PREFIX)


def is_recordable_utterance(query: str) -> bool:
    """是否把这句顾客话写入滚动档案。中控指令、空串、纯应和都不要。"""
    text = (query or "").strip()
    if not text or is_system_active_ask_query(text):
        return False
    if len(text) < 2:
        return False
    return _BACKCHANNEL.fullmatch(text) is None


def _char_bigrams(text: str) -> set:
    """去空白后的相邻二字集合，用于中文短句重合判断。"""
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def text_coverage_ratio(left: str, right: str) -> float:
    """
    较短一方的字 bigram 被较长一方覆盖的比例，范围 [0, 1]。

    比 Jaccard 更适合「短换皮句 vs 带称呼/解释的长问句」：
    长句多出来的寒暄不会把重合度稀释掉。
    """
    a = _char_bigrams(left)
    b = _char_bigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def text_overlap_ratio(left: str, right: str) -> float:
    """两段中文的字 bigram Jaccard 重合度，范围 [0, 1]。保留给调试/对照。"""
    a = _char_bigrams(left)
    b = _char_bigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _clause_cores(text: str) -> List[str]:
    """
    原文 + 带疑问线索的分句。

    不抽取槽位，只避免前后寒暄把覆盖度拉低。
    例如「刘先生，您平时…还是家里一起用呢？这能帮我推荐」会多出一个「还是…」分句。
    """
    text = (text or "").strip()
    if not text:
        return []
    cores = [text]
    for clause in _CLAUSE_SPLIT.split(text):
        clause = clause.strip(" ，,")
        if len(clause) < 4:
            continue
        if any(hint in clause for hint in _QUESTION_HINTS):
            cores.append(clause)
    return cores


def utterances_as_retrieval_hints(utterances: Sequence[str], limit: int = 4) -> str:
    """把最近几句顾客原话拼进 FAISS 查询，不依赖预定义标签。"""
    recent = [str(x).strip() for x in utterances if str(x).strip()][-limit:]
    if not recent:
        return ""
    return "顾客已说 " + " ".join(recent) + " 不要再问已经说过的内容"


def _max_coverage(left: str, right: str) -> float:
    """原文与问句核心两两取最大覆盖度。"""
    best = 0.0
    for a in _clause_cores(left):
        for b in _clause_cores(right):
            best = max(best, text_coverage_ratio(a, b))
    return best


def should_drop_retrieved_script(
    doc_text: str,
    last_active_ask: str = "",
    answered_asks: Sequence[str] = (),
    customer_utterances: Sequence[str] = (),
    question_coverage: float = RETRIEVAL_QUESTION_COVERAGE,
    utterance_coverage: float = RETRIEVAL_UTTERANCE_COVERAGE,
) -> bool:
    """
    检索到的主动话语术是否应丢弃。

    两条与主题无关的规则：
    - 和「上一句主动问 / 已经问过的原句」覆盖度够高 → 换皮同问，丢掉。
    - 和「顾客亲口说过的原话」覆盖度够高 → 话术在复述已答内容，丢掉。

    不解析话术在问谁开、预算还是颜色。任意主题都走同一条规则。
    """
    body = (doc_text or "").strip()
    if not body:
        return False

    for prev in [last_active_ask, *answered_asks]:
        prev = (prev or "").strip()
        if prev and _max_coverage(body, prev) >= question_coverage:
            return True

    for uttered in customer_utterances:
        uttered = (uttered or "").strip()
        if uttered and _max_coverage(body, uttered) >= utterance_coverage:
            return True
    return False


def answered_ask_texts(pairs: Sequence[Dict[str, Any]]) -> List[str]:
    """从 {ask, reply} 列表抽出曾经问过的原句，供检索过滤。"""
    out: List[str] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        ask = str(item.get("ask") or "").strip()
        if ask:
            out.append(ask)
    return out
