"""请求/响应模型：FastAPI自动做校验和文档，业务层不用手写参数检查。"""
from typing import List
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """POST /api/chat 请求体"""
    question: str = Field(..., min_length=1, max_length=500, description="用户问题，1~500字")

class ChatResponse(BaseModel):
    """POST /api/chat 响应体"""
    answer: str = Field(..., description="完整回答")
    route: str = Field(..., description="路由结果：direct / agent / none")
    tools_used: List[str] = Field(default_factory=list, description="本轮调用的工具列表")
    latency_sec: float = Field(0.0, description="总耗时（秒）")

class HealthResponse(BaseModel):
    """GET /api/health 响应体"""
    status: str = "ok"
    vector_store_ready: bool = Field(..., description="向量库是否就绪")
    llm_model: str = Field(..., description="当前使用的LLM模型名")
