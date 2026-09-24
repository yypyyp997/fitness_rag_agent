"""LangGraph 工作流编排。

之前的run_agent 是"万能但较重"的单体Agent：任何问题都要走完整的
思考-行动循环（选工具→调工具→再思考...），简单问题也要2次以上模型调用。

本模块在其上加一层 LangGraph StateGraph 编排，按问题难度分流：
- RAG直答支路（direct）：简单知识问答，检索一次 + 生成一次，不走Agent循环
- Agent支路（agent）：计划制定、数学计算、联网搜索等复杂任务，走CP6完整循环

LangGraph 三个核心概念在这里的落点：
- StateGraph(OrchestratorState)：定义状态结构，节点返回局部更新自动合并
- add_conditional_edges：路由节点按 route 字段动态选择下一个节点
- runtime context：Agent支路把 tools_used 从中间件带回编排层（提交点9评估用）
"""
from typing import TypedDict
import time

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agent.callbacks import TokenUsageHandler
from agent.middleware import postprocess_output, preprocess_input
from agent.react_agent import get_agent
from model.factory import get_llm
from prompts import get_prompt
from tools import rag_search
from utils import get_logger

logger = get_logger(__name__)

# 路由提示词：属于内部结构逻辑，不随业务调参，放代码常量（想调节再挪去prompts/目录）
_ROUTER_PROMPT = (
    "你是问题路由器，判断用户问题走哪条处理路径，只回答一个单词：\n"
    "- direct：简单健身知识问答，查本地知识库就能回答"
    "（动作要领、训练原理、饮食营养等单一知识点）\n"
    "- agent：复杂任务，包括制定训练计划、数学计算、多知识点综合、"
    "需要联网搜索、闲聊或与健身无关的问题\n"
    "只输出 direct 或 agent，不要任何解释。\n\n"
    "用户问题："
)

class OrchestratorState(TypedDict):
    """工作流状态：问题进，回答+路由+工具统计出"""
    question: str
    route: str
    answer: str
    tools_used: list

def route_question(state: OrchestratorState) -> dict:
    """路由节点：LLM判断走哪条支路，调用失败兜底走Agent（通用路径不会答不了）"""
    question = state["question"]
    try:
        resp = get_llm().invoke(_ROUTER_PROMPT + question)
        text = resp.content.strip().lower() if isinstance(resp.content, str) else ""
        route = "direct" if "direct" in text else "agent"
    except Exception as e:
        logger.warning(f"路由分类失败，兜底走Agent支路: {e}")
        route = "agent"
    logger.info(f"路由结果: {route}（问题: {question[:30]}）")
    return {"route": route}

def direct_rag_answer(state: OrchestratorState) -> dict:
    """RAG直答支路：检索 -> 按消融胜出提示词单次生成（无Agent循环，快且省token）"""
    question = state["question"]

    # 复用CP5的rag_search工具做检索（带来源标注的统一格式）
    context = rag_search.invoke({"query": question})

    # rag_prompt.md 里有 {context} 和 {question} 两个占位符
    prompt = get_prompt("rag_prompt").format(context=context, question=question)
    resp = get_llm().invoke(prompt)
    answer = resp.content.strip() if isinstance(resp.content, str) else str(resp.content)

    logger.info(f"RAG直答完成，回答长度{len(answer)}字")
    return {"answer": answer, "tools_used": ["rag_search"]}

def agent_answer(state: OrchestratorState) -> dict:
    """Agent支路：CP6完整思考-行动循环，并从runtime context回收工具使用统计"""
    question = preprocess_input(state["question"])
    if not question:
        return {"answer": "请输入问题", "tools_used": []}

    # 自建context字典传入：monitor_tool中间件会向它写入tools_used（按引用共享）
    ctx = {"tools_used": []}
    result = get_agent().invoke(
        {"messages": [HumanMessage(content=question)]},
        context=ctx,
        config={"callbacks": [TokenUsageHandler()]},
    )
    tools_used = ctx.get("tools_used", [])
    logger.info(f"Agent支路完成，本轮使用工具: {tools_used}")
    return {"answer": postprocess_output(result), "tools_used": tools_used}

def _route_by_state(state: OrchestratorState) -> str:
    """条件边路由函数：读state里的route字段决定去向"""
    return state.get("route", "agent")

_orchestrator_cache = None

def build_orchestrator():
    """构建编排工作流（全局只建一次）：
    START -> router -> (direct_answer 或 agent_answer) -> END
    """
    global _orchestrator_cache
    if _orchestrator_cache is None:
        graph = StateGraph(OrchestratorState)
        graph.add_node("router", route_question)
        graph.add_node("direct_answer", direct_rag_answer)
        graph.add_node("agent_answer", agent_answer)

        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            _route_by_state,
            {"direct": "direct_answer", "agent": "agent_answer"},
        )
        graph.add_edge("direct_answer", END)
        graph.add_edge("agent_answer", END)

        _orchestrator_cache = graph.compile()
        logger.info("LangGraph编排工作流构建完成")
    return _orchestrator_cache

def run_orchestrator(question: str) -> dict:
    """编排统一入口（提交点8的main.py、提交点9评估都调它）

    Returns:
        dict: {"answer": 最终回答, "route": 走的支路, "tools_used": 工具使用统计}
    """
    question = (question or "").strip()
    if not question:
        return {"answer": "请输入问题", "route": "none", "tools_used": []}

    result = build_orchestrator().invoke({"question": question})
    logger.info(
        f"编排完成：路由={result.get('route')} "
        f"工具={result.get('tools_used')} 回答{len(result.get('answer', ''))}字"
    )
    return {
        "answer": result.get("answer", ""),
        "route": result.get("route", ""),
        "tools_used": result.get("tools_used", []),
    }

def _stream_dashscope(prompt: str):
    """用dashscope原生SDK流式生成（增量输出）。

    为什么不用ChatTongyi.stream：实测它把整个流聚合成1块吐出（250字回答
    只返回1个chunk），界面看不到打字机效果；原生SDK同问题返回52个增量块。

    Yields:
        str: 增量文本块
    """
    import dashscope
    from config import settings

    resp = dashscope.Generation.call(
        model=settings.LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        result_format="message",
        stream=True,
        incremental_output=True,
        api_key=settings.DASHSCOPE_API_KEY,
    )
    for chunk in resp:
        try:
            delta = chunk.output.choices[0].message.content or ""
        except Exception:
            delta = ""
        if delta:
            yield delta

def stream_orchestrator(question: str):
    """流式编排入口：逐步产出回答文本块，最后产出一个元信息dict。

    与 run_orchestrator 的分工：
    - run_orchestrator：阻塞式拿完整结果（命令行/评估用，走LangGraph图）
    - stream_orchestrator：边生成边输出（Streamlit界面用，体验优先）

    用法：
        for chunk in stream_orchestrator(q):
            if isinstance(chunk, dict):
                meta = chunk   # 最后一个产出：含 route / tools_used / answer
            else:
                show(chunk)    # 文本块，逐步拼接展示

    说明：两条支路都改用dashscope原生流式（ChatTongyi.stream实测会整块聚合）；
    流式路径不做token统计（评估模块走非流式路径，token数据完整）。
    """
    question = (question or "").strip()
    if not question:
        yield {"answer": "请输入问题", "route": "none", "tools_used": []}
        return

    # 路由判断本身不流式（输出只有一个单词，流式无意义），约0.5秒
    route = route_question({"question": question})["route"]

    if route == "direct":
        # RAG直答支路：检索一次 + dashscope原生流式生成
        context = rag_search.invoke({"query": question})
        prompt = get_prompt("rag_prompt").format(context=context, question=question)
        parts = []
        for delta in _stream_dashscope(prompt):
            parts.append(delta)
            yield delta
        yield {"route": "direct", "tools_used": ["rag_search"], "answer": "".join(parts)}
    else:
        # Agent支路：先完整跑Agent循环（拿工具统计和最终回答），再把最终回答
        # 按小段切分逐步吐出——工具调用轮次本来就没有可流式展示的正文
        result = run_orchestrator(question)
        answer = result["answer"]
        # 按标点切句渐进展示，每块约20字，模拟真实生成节奏
        step = 20
        for i in range(0, len(answer), step):
            piece = answer[i:i + step]
            yield piece
            time.sleep(0.03)
        yield {
            "route": result["route"],
            "tools_used": result["tools_used"],
            "answer": answer,
        }
