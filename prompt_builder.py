from langchain_core.prompts import ChatPromptTemplate

def build_prompt(query: str, context: str) -> str:
    """
    组装 RAG 提示词字符串，融合状态判断指令。
    要求模型：
    1. 基于【资料】回答问题，资料不足则说明“资料不足”。
    2. 根据用户问题判断意图状态，并在回答末尾附加 <INTENT>状态</INTENT>。
    3. 状态可选值：CUSTOMER_ENTER, NEEDS_GUIDANCE, MOVE_TO_WAIT, WAIT_FOR_TALK（默认）。
    """
    template = (
        "你是一个智能导购助手。请严格遵循以下要求：\n"
        "1. 基于【资料】回答问题。如果资料不足以回答问题，请明确回复“抱歉，我无法回答你的问题”。\n"
        "2. 同时你要根据根据用户的输入，判断用户想要进入哪个状态，并在回答末尾附加状态标签，格式为：<INTENT>状态</INTENT>\n"
        "   可用的状态及示例：\n"
        "   - CUSTOMER_ENTER：客户进店（例如：“有人在吗”）\n"
        "   - NEEDS_GUIDANCE：需要引导参观（例如：“参观一下”、“带我看看”、“带我逛逛”、“想看看”）\n"
        "   - MOVE_TO_WAIT：退下/不用了（例如：“退一下”、“不用了”、“退下”）\n"
        "   - WAIT_FOR_TALK：等待对话（默认状态，用于普通对话或不确定的情况）\n"
        "3. 如果不确定用户意图，则输出 WAIT_FOR_TALK 状态。\n"
        "4. 回答应简洁、可追溯，并严格以状态标签结尾（标签后不要添加其他内容）。\n"
        "\n"
        "【资料】\n"
        "{context}\n"
        "\n"
        "【问题】\n"
        "{question}\n"
        "\n"
        "请输出回答（末尾必须包含 <INTENT>状态</INTENT>）："
    )
    prompt = ChatPromptTemplate.from_template(template)
    return prompt.format(context=context, question=query)