# api/main.py
"""FastAPI服务层：把编排层封装成HTTP接口。

设计原则：
- api层只做协议转换（HTTP/SSE <- 函数调用），业务逻辑全部留在agent/rag/tools层
- lifespan里预热向量库和编排工作流，避免首个请求背上1~2分钟的建库等待
- /api/chat/stream 用SSE：与前端fetch/EventSource天然兼容，也是通义千问
  原生流式的标准Web出口（OpenAI/Anthropic的流式API都用这个协议）

启动（项目根目录）：
    uvicorn api.main:app --host 0.0.0.0 --port 8000

交互式文档（自动生成）：
    http://localhost:8000/docs
"""
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest, ChatResponse, HealthResponse
from config import settings
from utils import get_logger

logger = get_logger(__name__)
# 请求访问日志单独写入 logs/api.log（logs/ 目录已在 .gitignore 中）
request_logger = get_logger("api.request", log_file="api.log")

# 应用状态（lifespan里填充）
_state = {"vector_store_ready": False}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预热：建/载向量库 + 构建编排图（首个请求因此不背冷启动）"""
    import os
    logger.info("服务启动：开始预热（向量库 + 编排工作流）...")
    if os.path.exists(settings.CHROMA_PERSIST_DIR) and os.listdir(settings.CHROMA_PERSIST_DIR):
        from rag import get_vector_store
        get_vector_store()
        _state["vector_store_ready"] = True
    else:
        # 提交点3：镜像不含预建向量库（.dockerignore 排除 chroma_db），启动时自动入库
        # 构建失败（如缺 DASHSCOPE_API_KEY）会直接抛出、容器快速失败——知识库建不出来时 RAG 服务没有启动价值
        logger.info("向量库缺失，启动时自动构建：加载 raw 知识库 → 切分 → DashScope Embedding 入库（约1-2分钟）...")
        from rag import prepare_chunks, build_vector_store
        build_vector_store(prepare_chunks())
        logger.info("向量库自动构建完成")
    # 预构建编排图（路由+双支路注册，不调API）
    from agent import build_orchestrator
    build_orchestrator()
    _state["vector_store_ready"] = (
        os.path.exists(settings.CHROMA_PERSIST_DIR) and bool(os.listdir(settings.CHROMA_PERSIST_DIR))
    )
    logger.info("预热完成，服务就绪")
    yield
    logger.info("服务关闭")

app = FastAPI(
    title="Fitness RAG Agent API",
    description="健身领域RAG智能体：知识问答（RAG直答）+ 复杂任务（ReAct Agent）双支路服务",
    version="1.1.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def log_request_middleware(request: Request, call_next):
    """请求日志中间件：记录方法、路径、状态码、耗时，写入 logs/api.log。"""
    start = time.perf_counter()
    response = await call_next(request)
    cost_ms = round((time.perf_counter() - start) * 1000, 2)
    request_logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} cost={cost_ms}ms"
    )
    return response

@app.get("/api/health", response_model=HealthResponse)
def health():
    """健康检查：探活 + 核心依赖就绪状态"""
    return HealthResponse(
        status="ok",
        vector_store_ready=_state["vector_store_ready"],
        llm_model=settings.LLM_MODEL_NAME,
    )

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """非流式问答：一次性返回完整结果。

    session_id 传入即启用多轮记忆（同会话可追问）；不传为单轮问答。
    """
    from agent import run_orchestrator

    start = time.perf_counter()
    try:
        result = run_orchestrator(req.question, session_id=req.session_id)
    except Exception as e:
        logger.error(f"编排执行失败: {e}")
        raise HTTPException(status_code=502, detail=f"上游模型/检索服务异常: {e}") from e
    latency = round(time.perf_counter() - start, 2)
    return ChatResponse(
        answer=result["answer"],
        route=result["route"],
        tools_used=result["tools_used"],
        latency_sec=latency,
    )

@app.delete("/api/sessions/{session_id}")
def clear_session(session_id: str):
    """清空指定会话的多轮记忆（前端"清空对话"按钮调它）"""
    from agent import memory
    deleted = memory.clear_session(session_id)
    return {"session_id": session_id, "deleted": deleted}

@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """流式问答（SSE协议），支持session_id多轮记忆。

    事件格式（text/event-stream）：
        event: delta   data: {"text": "增量文本"}      （多次，逐块）
        event: meta    data: {"route": ..., "tools_used": [...], "session_id": ...}（最后1次）
        event: done    data: [DONE]
    """
    from agent import stream_orchestrator

    def event_gen():
        start = time.perf_counter()
        try:
            for chunk in stream_orchestrator(req.question, session_id=req.session_id):
                if isinstance(chunk, dict):
                    meta = {**chunk, "latency_sec": round(time.perf_counter() - start, 2),
                            "session_id": req.session_id}
                    yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"
                else:
                    yield f"event: delta\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            logger.error(f"流式执行失败: {e}")
            err = {"error": str(e)}
            yield f"event: error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 防nginx等反代缓冲SSE
        },
    )
