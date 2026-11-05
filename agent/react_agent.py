"""

create_agent 是 langchain 1.x 官方推荐入口，替代旧的 create_react_agent：
- middleware: @wrap_tool_call / @before_model / @dynamic_prompt 三件套
- context_schema: 定义 runtime.context 的结构（工具使用统计）
- run_agent(): 对外统一入口，main.py / app.py / 评估脚本都调它
"""
from typing import TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from agent.callbacks import TokenUsageHandler
from agent.middleware import (
    log_before_model,
    monitor_tool,
    postprocess_output,
    preprocess_input,
    system_prompt_switch,
)
from model.factory import get_llm
from prompts import get_prompt
from tools import get_tools
from utils import get_logger

logger = get_logger(__name__)

_agent_cache = None

#Agent 单次运行步数预算：N 轮工具 + 收尾作答 = 2N+2 个图节点步（langgraph 1.2.11 实测）。
# 16 = 至多 7 轮工具 + 作答：高于历史最大值（5 连查，见踩坑 9.15）留 2 轮余量；
# 超限抛 GraphRecursionError（Exception 子类），由编排层防线3 捕获降级，调用方不见异常。
AGENT_RECURSION_LIMIT = 16

class AgentContext(TypedDict, total=False):
    """runtime上下文结构：工具使用统计由monitor_tool写入，编排层和评估模块可读取"""
    tools_used: list

def build_system_prompt() -> str:
    """标准模式系统提示词（调试/评估用；运行时由system_prompt_switch动态决定）"""
    return get_prompt("agent_system") + "\n\n" + get_prompt("tool_desc")

def get_agent():
    """构建智能体：通义千问 + 工具三件套 + 中间件三件套（全局只建一次）"""
    global _agent_cache
    if _agent_cache is None:
        _agent_cache = create_agent(
            model=get_llm(),
            tools=get_tools(),
            middleware=[monitor_tool, log_before_model, system_prompt_switch],
            context_schema=AgentContext,
        )
        logger.info("Agent构建完成（官方middleware版）")
    return _agent_cache

def run_agent(question: str) -> str:
    """单轮对话入口：预处理 -> Agent执行 -> 后处理

    Args:
        question: 用户问题

    Returns:
        str: Agent的最终回答
    """
    question = preprocess_input(question)
    if not question:
        return "请输入问题"

    agent = get_agent()
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        context={"tools_used": []},
        config={"recursion_limit": AGENT_RECURSION_LIMIT, "callbacks": [TokenUsageHandler()]},
    )
    answer = postprocess_output(result)
    logger.info(f"Agent回答完成，长度{len(answer)}字")
    return answer
