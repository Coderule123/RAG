from typing import Optional

from langchain_core.prompts import ChatPromptTemplate


def build_prompt(
    query: str, context: str, vision_user_id: Optional[str] = None, vioce_user_id: Optional[str] = None
) -> str:
    """
    组装 RAG 提示词字符串，融合状态判断指令。
    要求模型：
    1. 基于【资料】回答问题，资料不足则说明“资料不足”。
    2. 根据用户问题判断意图状态，并在回答末尾附加 <INTENT>状态</INTENT>。
    3. 状态可选值：CUSTOMER_ENTER, NEEDS_GUIDANCE, MOVE_TO_WAIT, WAIT_FOR_TALK（默认）。
    """
    user_line = ""

    # 处理视觉用户姓名
    if vision_user_id is not None and str(vision_user_id).strip():
        user_line += f"当前看到的用户的姓名是：{str(vision_user_id).strip()}\n"

    # 处理语音用户 ID
    if vioce_user_id is not None and str(vioce_user_id).strip():
        user_line += f"当前听到的用户的姓名是：{str(vioce_user_id).strip()}\n"

    # 保留原结尾的双换行（可根据实际需要调整）
    if user_line:   # 只有在至少添加了一行内容时才追加换行
        user_line += "\n"

    template = (
        "你是一个智能导购助手。请严格遵循以下要求：\n"
        "基于【资料】回答问题。如果资料不足以回答问题，请明确回复“抱歉，我无法回答你的问题”。\n"
        "\n"
        "{user_line}"
        "【资料】\n"
        "{context}\n"
        "\n"
        "【问题】\n"
        "{question}\n"
        "\n"
        "请输出回答（末尾必须包含 <INTENT>状态</INTENT>）："
    )
    prompt = ChatPromptTemplate.from_template(template)
    return prompt.format(user_line=user_line, context=context, question=query)