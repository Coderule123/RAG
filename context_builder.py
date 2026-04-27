from typing import Dict, List


def build_context(docs: List[Dict], vector_threshold: float = 0.0) -> str:
    """
    将重排后的片段拼成一段可读上下文，带来源、页码、向量分与重排分便于人工核对。
    供下游 LLM 使用时可按需在 prompt 中裁剪。
    """
    lines = []
    for idx, doc in enumerate(docs, start=1):
        vector_score = doc.get('vector_score', 0.0)
        if vector_score is None:
            vector_score = 0.0
        if vector_score >= vector_threshold:
            text = doc.get('text', '')
            if text:  # 只添加非空文本
                lines.append(text)
    return "\n\n".join(lines)
