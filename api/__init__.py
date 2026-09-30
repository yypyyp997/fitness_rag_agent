"""api包：FastAPI服务层。

与agent编排层的分工：
- agent/：业务核心（LangGraph编排、RAG、工具），不知道任何Web框架的存在
- api/：Web壳（HTTP路由、SSE、错误处理），只做协议转换，不写业务逻辑

启动方式（项目根目录）：
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
