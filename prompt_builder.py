from langchain_core.prompts import ChatPromptTemplate


def build_prompt(query: str, context: str) -> str:
    """
    仅组装 RAG 提示词字符串，不调用大模型；由外部对话模块接 LLM。
    """
    template = (
        "你是一个问答助手。请基于给定资料回答问题。"
        "如果资料不足，请明确说明“资料不足”。\n\n"
        "【资料】\n{context}\n\n"
        "【问题】\n{question}\n\n"
        "请输出简洁且可追溯的答案。"
    )
    prompt = ChatPromptTemplate.from_template(template)
    return prompt.format(context=context, question=query)
