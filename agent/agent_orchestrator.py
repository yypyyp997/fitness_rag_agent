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

多轮记忆（P0迭代）：session_id 可选参数——传入即注入最近6条历史到
路由/直答/Agent循环三处，并落库；不传则与单轮版行为完全一致。

故障兜底设计（对应 README 踩坑 9.17）：
整个编排层遵循一条总原则——「异常可以向上传播，但每一层只能有一个裸奔口，
且编排统一入口绝不上抛」。此前三条故障路径会直接炸穿调用方：
- direct 支路的检索/生成裸调用（无 try 保护）
- agent 支路 rag_search 异常穿透 ToolNode（默认策略只兜参数校验错）
- 落库块无保护（回答生成失败时 user 消息也不会落库）
加固后防线分层：rag_search 工具内捕获（防线1）→ 节点级 try/except（防线2/3）
→ 落库原子化（防线4）→ run_orchestrator 总兜底（防线5）。任何一层失守，
下一层接住；即使向量库目录被删、API Key 失效，入口永远返回 dict、绝不 raise。
"""
from typing import TypedDict
import time

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agent.callbacks import TokenUsageHandler
from agent.memory import format_history_block, get_recent_messages
from agent.middleware import (
    needs_plan_retry,
    postprocess_output,
    preprocess_input,
)
from agent.react_agent import get_agent
from model.factory import get_llm
from prompts import get_prompt
from tools import rag_search
from tools.rag_search import RETRIEVAL_DEGRADED_HINT
from utils import get_logger

logger = get_logger(__name__)

# 路由提示词：属于内部结构逻辑，不随业务调参，放代码常量（想调参再挪去prompts/目录）
# 注：分类边界由提交点9评估驱动优化——问具体数字/怎么办的知识问题易被误判为agent
_ROUTER_PROMPT = (
    "你是问题路由器，判断用户问题走哪条处理路径，只回答一个单词：\n"
    "- direct：健身知识问答，答案能从本地知识库查到。包括：动作要领、训练原理、"
    "饮食营养（含具体数字，如摄入量、热量缺口、休息时长）、伤病处理与建议、"
    "怎么办/为什么/有什么类问题\n"
    "- agent：需要执行任务的复杂请求。包括：数学计算（含算式）、制定训练计划、"
    "多知识点综合方案、联网搜索、闲聊或与健身无关\n"
    "只输出 direct 或 agent，不要任何解释。\n\n"
    "用户问题："
)

# 计划护栏重试提醒：重试轮的最后一轮 Human 消息决定 dynamic_prompt 是否切计划模式，
# 文案必须命中 _PLAN_KEYWORDS（见 README 9.15），此契约由单测守护
PLAN_RETRY_REMINDER = (
    "这是增肌计划类问题。上一版回答缺少组数×次数，"
    "请重新输出完整的增肌计划：必须包含每周训练安排，"
    "每个动作标注组数×次数，并附注意事项。"
)

# 各防线降级文案：保证「任何情况下都有回答」，同时把故障类型透出给用户
_DEGRADED_RETRIEVAL = "抱歉，本地知识库暂时不可用（向量库可能缺失或损坏），请先运行 python main.py 重建向量库后再试。"
_DEGRADED_GENERATION = "抱歉，模型服务暂时不可用（网络波动或配额限流），请稍后重试。"

# 降级文案登记表：用于区分「真回答」与「降级回答」（降级回答不落库，避免污染多轮历史）
_DEGRADED_ANSWERS = {_DEGRADED_RETRIEVAL, _DEGRADED_GENERATION}


def _is_degraded(answer: str) -> bool:
    """判断回答是否为防线降级文案（降级回答不写入多轮记忆）"""
    return answer in _DEGRADED_ANSWERS


def _retrieval_failure(exc: Exception) -> bool:
    """粗分异常来源：检索层（连接/初始化错误）→ True，生成层（限流/鉴权）→ False。

    只服务流式路径的降级文案选择；非流式路径在 direct 节点内分段捕获，
    不需要这个猜测函数。误判的代价只是文案不同，不影响可用性。
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    for kw in ("chroma", "sqlite", "retriev", "dimension", "embedding", "connection"):
        if kw in text:
            return True
    return False


class OrchestratorState(TypedDict):
    """工作流状态：问题进，回答+路由+工具统计出（history为多轮记忆注入，可选）"""
    question: str
    route: str
    answer: str
    tools_used: list
    history: list

def route_question(state: OrchestratorState) -> dict:
    """路由节点：LLM判断走哪条支路，调用失败兜底走Agent（通用路径不会答不了）"""
    question = state["question"]
    # 多轮上下文：追问（如"那硬拉呢？"）单独看无法路由，把近期对话带入判定
    history = state.get("history") or []
    route_prompt = format_history_block(history) + _ROUTER_PROMPT if history else _ROUTER_PROMPT
    try:
        resp = get_llm().invoke(route_prompt + question)
        text = resp.content.strip().lower() if isinstance(resp.content, str) else ""
        route = "direct" if "direct" in text else "agent"
    except Exception as e:
        logger.warning(f"路由分类失败，兜底走Agent支路: {e}")
        route = "agent"
    logger.info(f"路由结果: {route}（问题: {question[:30]}）")
    return {"route": route}

def direct_rag_answer(state: OrchestratorState) -> dict:
    """RAG直答支路：检索 -> 按消融胜出提示词单次生成（无Agent循环，快且省token）

    防线2（节点级）：检索失败与生成失败分别降级，节点永不抛异常。
    此前本节点是裸奔口——rag_search 或 get_llm().invoke 任何一处抛异常
    都会穿出节点炸穿整场（探针实证 During task with name 'direct_answer'）。
    """
    question = state["question"]
    # 历史块拼在检索模板前：回答既要基于检索内容，也要衔接上文（"那减脂期呢"）
    history_block = format_history_block(state.get("history") or [])

    # 防线2a：检索失败降级。两种识别途径：
    # ① 工具抛异常（未被防线1覆盖的场景）→ except 接住；
    # ② 工具内部已捕获但返回降级文案（防线1）→ 短路生成，不再浪费一次必然无意义的模型调用。
    try:
        context = rag_search.invoke({"query": question})
    except Exception as e:
        logger.error(f"RAG直答检索失败，降级返回提示: {e}")
        return {"answer": _DEGRADED_RETRIEVAL, "tools_used": ["rag_search"]}
    if context == RETRIEVAL_DEGRADED_HINT:
        logger.info("RAG直答：检索层已降级（防线1），短路生成直接返回提示")
        return {"answer": _DEGRADED_RETRIEVAL, "tools_used": ["rag_search"]}

    # rag_prompt.md 里有 {context} 和 {question} 两个占位符
    prompt = get_prompt("rag_prompt").format(context=context, question=question)
    # 防线2b：生成失败降级（429/网络/超时等，LLM内部重试耗尽后走到这里）
    try:
        resp = get_llm().invoke(history_block + prompt if history_block else prompt)
    except Exception as e:
        logger.error(f"RAG直答生成失败，降级返回提示: {e}")
        return {
            "answer": _DEGRADED_GENERATION,
            "tools_used": ["rag_search"],
        }
    answer = resp.content.strip() if isinstance(resp.content, str) else str(resp.content)

    logger.info(f"RAG直答完成，回答长度{len(answer)}字")
    return {"answer": answer, "tools_used": ["rag_search"]}

def agent_answer(state: OrchestratorState) -> dict:
    """Agent支路：CP6完整思考-行动循环，并从runtime context回收工具使用统计

    末尾接一层计划模式护栏：若计划类问题的回答漏掉「组数×次数」，
    追加一句提醒重跑一次；重试结果无论好坏直接采用，最多一次，避免死循环。

    防线3（节点级）：Agent循环整体捕获，失败降级为提示文案，节点永不抛异常。
    此前本节点内模型调用失败（如401/重试耗尽）会穿出节点炸穿整场
    （探针实证 During task with name 'model' -> 'agent_answer'）。
    """
    question = preprocess_input(state["question"])
    if not question:
        return {"answer": "请输入问题", "tools_used": []}

    # 历史转成消息列表：Agent循环里模型能看到上文（Human/AIMessage交替）
    from langchain_core.messages import AIMessage
    history_messages = []
    for m in (state.get("history") or []):
        if m["role"] == "user":
            history_messages.append(HumanMessage(content=m["content"]))
        else:
            history_messages.append(AIMessage(content=m["content"]))

    try:
        # 自建context字典传入：monitor_tool中间件会向它写入tools_used（按引用共享）
        ctx = {"tools_used": []}
        result = get_agent().invoke(
            {"messages": history_messages + [HumanMessage(content=question)]},
            context=ctx,
            config={"callbacks": [TokenUsageHandler()]},
        )
        tools_used = ctx.get("tools_used", [])
        answer = postprocess_output(result)

        # 计划模式运行时护栏：命中则带一句提醒重跑一次（最多1次，结果直接采用）
        if needs_plan_retry(question, answer):
            logger.warning(
                f"计划护栏触发（问题: {question[:30]}），追加提醒重跑一次"
            )
            reminder = PLAN_RETRY_REMINDER
            retry_ctx = {"tools_used": []}
            retry_result = get_agent().invoke(
                {
                    "messages": history_messages
                    + [
                        HumanMessage(content=question),
                        AIMessage(content=answer),
                        HumanMessage(content=reminder),
                    ]
                },
                context=retry_ctx,
                config={"callbacks": [TokenUsageHandler()]},
            )
            retry_tools_used = retry_ctx.get("tools_used", [])
            if retry_tools_used:
                tools_used = retry_tools_used
            answer = postprocess_output(retry_result)
    except Exception as e:
        logger.error(f"Agent支路执行失败，降级返回提示: {e}")
        return {"answer": _DEGRADED_GENERATION, "tools_used": []}

    logger.info(f"Agent支路完成，本轮使用工具: {tools_used}")
    return {"answer": answer, "tools_used": tools_used}

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

def _save_history(session_id: str | None, question: str, result: dict) -> None:
    """多轮记忆落库（原子化，防线4）：user+assistant 两条要么都写、要么都不写。

    此前落库块裸奔：append user 成功后 append assistant 失败会留下只有
    user 的残缺轮次；整个落库块不保护还会让「回答已成功但落库失败」
    把异常抛给调用方，前面所有防线白防。任何落库异常在这里吞掉并记日志。
    """
    if not session_id:
        return
    try:
        from agent import memory
        memory.append_message(session_id, "user", question)
        memory.append_message(
            session_id, "assistant", result.get("answer", ""),
            route=result.get("route"), tools_used=result.get("tools_used"),
        )
    except Exception as e:
        logger.error(f"多轮记忆落库失败（本轮对话不入历史）: {e}")

def run_orchestrator(question: str, session_id: str | None = None) -> dict:
    """编排统一入口（提交点8的main.py、提交点9评估都调它）

    兼容说明：session_id 为 None 时行为与单轮版完全一致（不读写记忆库），
    main.py / evaluation 等旧调用方零改动。

    防线5（总兜底）：图执行意外失败时返回降级 dict 而非向上抛异常——
    调用方（main/app/api/eval）从此不需要处理「编排层炸穿」的情况。
    评估器兼容：降级 dict 含 route/tools_used/answer 字段，评估三指标
    照常判定（降级回答通常挂关键词，记 FAIL 而不是 ERROR，见 README 9.17）。

    Returns:
        dict: {"answer": 最终回答, "route": 走的支路, "tools_used": 工具使用统计}
    """
    question = (question or "").strip()
    if not question:
        return {"answer": "请输入问题", "route": "none", "tools_used": []}

    # 多轮记忆：有session_id时取最近6条历史注入三处（路由/直答/Agent循环），
    # 并在回答成功后落库（失败不污染历史）
    try:
        from agent import memory
        history = memory.get_recent_messages(session_id) if session_id else []
    except Exception as e:
        logger.error(f"读取多轮历史失败（本轮按无历史继续）: {e}")
        history = []

    try:
        result = build_orchestrator().invoke({"question": question, "history": history})
        if not isinstance(result, dict):
            raise TypeError(f"图执行返回了非dict结果: {type(result).__name__}")
        payload = {
            "answer": result.get("answer", ""),
            "route": result.get("route", ""),
            "tools_used": result.get("tools_used", []),
        }
    except Exception as e:
        # 防线5：图内意外异常（理论上前四层已兜住，这里是最后一道保险）
        logger.error(f"编排图执行意外失败，返回降级回答: {e}")
        payload = {"answer": _DEGRADED_GENERATION, "route": "agent", "tools_used": []}

    # 落库走防线4的原子化函数：只对真实完成的回答落库（降级回答不污染历史）
    if session_id and not _is_degraded(payload.get("answer", "")):
        _save_history(session_id, question, payload)

    logger.info(
        f"编排完成：路由={payload.get('route')} "
        f"工具={payload.get('tools_used')} 回答{len(payload.get('answer', ''))}字"
    )
    return payload

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

def stream_orchestrator(question: str, session_id: str | None = None):
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

    故障兜底（防线1/2/5在流式路径的映射）：与 run_orchestrator 同一原则——
    永远产出「文本块 + 收尾meta dict」，绝不向生成器外抛异常。
    """
    question = (question or "").strip()
    if not question:
        yield {"answer": "请输入问题", "route": "none", "tools_used": []}
        return

    # 路由判断本身不流式（输出只有一个单词，流式无意义），约0.5秒
    from agent import memory
    history = memory.get_recent_messages(session_id) if session_id else []
    route = route_question({"question": question, "history": history})["route"]

    if route == "direct":
        # RAG直答支路：检索一次 + dashscope原生流式生成（与非流式防线同构：
        # 检索隔离 → 降级短路 → 生成隔离）
        try:
            context = rag_search.invoke({"query": question})
        except Exception as e:
            logger.error(f"流式直答检索失败，降级返回提示: {e}")
            context = RETRIEVAL_DEGRADED_HINT

        if context == RETRIEVAL_DEGRADED_HINT:
            # 检索层已失败（防线1降级或异常兜底）：短路流式生成，直接产出检索降级文案
            answer = _DEGRADED_RETRIEVAL
            yield answer
        else:
            try:
                prompt = get_prompt("rag_prompt").format(context=context, question=question)
                history_block = format_history_block(history)
                if history_block:
                    prompt = history_block + prompt
                parts = []
                for delta in _stream_dashscope(prompt):
                    parts.append(delta)
                    yield delta
                answer = "".join(parts)
            except Exception as e:
                logger.error(f"流式直答生成失败，降级返回提示: {e}")
                answer = _DEGRADED_GENERATION
                yield answer
        if session_id and answer and not _is_degraded(answer):
            memory.append_message(session_id, "user", question)
            memory.append_message(session_id, "assistant", answer,
                                  route="direct", tools_used=["rag_search"])
        yield {"route": "direct", "tools_used": ["rag_search"], "answer": answer}
    else:
        # Agent支路：先完整跑Agent循环（拿工具统计和最终回答），再把最终回答
        # 按小段切分逐步吐出——工具调用轮次本来就没有可流式展示的正文
        try:
            result = run_orchestrator(question, session_id=session_id)
        except Exception as e:
            # 理论上防线5已兜住，这里防意外（含Agent支路切块逻辑前的未知异常）
            logger.error(f"流式Agent支路意外失败: {e}")
            result = {"answer": _DEGRADED_GENERATION, "route": "agent", "tools_used": []}
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
