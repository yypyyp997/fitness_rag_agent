"""Agent中间件（官方middleware API版）：
- @wrap_tool_call  工具执行监控：可拦截/改写，并把使用统计写进runtime上下文
- @before_model    模型调用前日志
- @dynamic_prompt  动态系统提示词：计划类问题自动切换计划模式
另含两个普通函数：输入预处理 / 输出提取
"""
from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import (
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime

from prompts import get_prompt
from utils import get_logger

logger = get_logger(__name__)

# 单次提问最大长度，防止误粘贴大段文本把token打爆
MAX_QUESTION_LEN = 500

# 计划类问题关键词：命中则切换计划模式提示词（想调行为改这里即可）
_PLAN_KEYWORDS = ("训练计划", "健身计划", "增肌计划", "减脂计划", "课表", "帮我制定", "帮我设计", "一周")

# 计划模式附加指令：拼在标准系统提示词后面，约束输出结构
_PLAN_MODE_SUFFIX = """
【计划模式】当前用户需要训练计划，回答必须包含：
1. 每周训练安排：训练日、目标肌群、动作、组数×次数（表格或分点）
2. 动作要领先调用 rag_search 查询知识库，基于检索内容描述
3. 训练量、热量等数字计算调用 calculator，禁止心算
4. 结尾附注意事项：恢复、饮食、循序渐进原则
"""

@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable,
) -> ToolMessage:
    """工具执行监控：记录名称/参数/结果，并把使用统计写进runtime上下文"""
    name = request.tool_call["name"]
    args = request.tool_call["args"]
    logger.info(f"——> 工具调用 {name} 参数: {args}")
    try:
        result = handler(request)
        logger.info(f"<—— 工具 {name} 执行成功")
        # 工具使用统计写入runtime上下文（提交点7编排层、提交点9评估可读取）
        ctx = getattr(request.runtime, "context", None)
        if ctx is not None:
            used = ctx.get("tools_used", [])
            used.append(name)
            ctx["tools_used"] = used
            logger.info(f"本轮已用工具: {used}")
        return result
    except Exception as e:
        logger.error(f"工具 {name} 执行失败，原因: {str(e)}")
        raise

@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    """模型调用前日志：输出当前消息数，复盘时能看出思考-行动循环了几轮"""
    logger.info(f"——> 即将调用模型，当前{len(state['messages'])}条消息")
    return None

@dynamic_prompt
def system_prompt_switch(request: ModelRequest) -> str:
    """动态系统提示词：计划类问题切换计划模式，其余用标准模式"""
    # 倒序找最近一条用户消息
    last_human = ""
    for msg in reversed(request.state["messages"]):
        if getattr(msg, "type", "") == "human":
            content = msg.content
            last_human = content if isinstance(content, str) else str(content)
            break

    base = get_prompt("agent_system") + "\n\n" + get_prompt("tool_desc")
    if any(kw in last_human for kw in _PLAN_KEYWORDS):
        logger.info("检测到计划类问题，切换【计划模式】提示词")
        return base + "\n" + _PLAN_MODE_SUFFIX
    return base

def preprocess_input(question: str) -> str:
    """输入中间件：去首尾空白、超长截断、空输入拦截"""
    q = (question or "").strip()
    if not q:
        logger.warning("收到空输入，已拦截")
        return ""
    if len(q) > MAX_QUESTION_LEN:
        q = q[:MAX_QUESTION_LEN]
        logger.warning(f"输入超过{MAX_QUESTION_LEN}字，已截断")
    return q

def postprocess_output(result: dict) -> str:
    """输出中间件：从Agent返回的messages里倒序找最终回答"""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai" and getattr(msg, "content", ""):
            text = msg.content
            if isinstance(text, str) and text.strip():
                return text.strip()
    return "Agent没有生成有效回答"
