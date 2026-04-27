from typing import Dict, List


def build_context(docs: List[Dict]) -> str:
    """
    将重排后的片段拼成一段可读上下文，带来源、页码、向量分与重排分便于人工核对。
    供下游 LLM 使用时可按需在 prompt 中裁剪。
    """
    lines = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", meta.get("page_number", "N/A"))
        chunk_id = meta.get("chunk_id", "N/A")
        lines.append(
            f"[片段{idx}] source={source} page={page} chunk={chunk_id} "
            f"vec={doc.get('vector_score', 0.0):.4f} rerank={doc.get('rerank_score', 0.0):.4f}\n"
            f"{doc.get('text', '')}"
        )
    return "\n\n".join(lines)
