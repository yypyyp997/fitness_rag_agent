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
【计划模式】当前用户需要训练计划，按以下优先级执行：
1. 最终回答第一部分必须是「每周训练安排」：训练日、目标肌群、动作、组数×次数。组数×次数由你基于渐进超负荷原则直接给出——这是专业推演，不是编造，不受「严格基于检索内容」限制
2. 检索预算：最多调用 rag_search 2 次。第 1 次查训练计划设计原则；第 2 次可选，只查你最不确定的单个动作。禁止逐个动作轮流查询
3. 动作要领与注意事项基于检索内容描述；训练量、热量等数字计算调用 calculator，禁止心算
4. 结尾附注意事项：恢复、饮食、循序渐进
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

def needs_plan_retry(question: str, answer: str) -> bool:
    """计划模式输出兜底护栏（纯函数，无副作用、不抛异常）：

    仅当 question 命中 _PLAN_KEYWORDS 时启用，满足任一条件返回 True：
      1. answer 为空白
      2. answer 以 postprocess_output 的失败占位符「Agent没有生成有效回答」开头
      3. answer 中不含「组」字（漏掉组数×次数）

    其余情况一律 False。非计划类问题永远不触发，避免误伤普通问答。
    任何异常都吞掉并返回 False，护栏本身不能成为新的故障点。
    """
    try:
        q = question if isinstance(question, str) else ""
        a = answer if isinstance(answer, str) else ""
        if not any(kw in q for kw in _PLAN_KEYWORDS):
            return False
        stripped = a.strip()
        if not stripped:
            return True
        if stripped.startswith("Agent没有生成有效回答"):
            return True
        return "组" not in a
    except Exception:
        return False
