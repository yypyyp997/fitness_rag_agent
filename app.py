import os
import time

import streamlit as st

# set_page_config 必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="健身知识问答 Agent",
    page_icon="💪",
    layout="centered",
)

from config import settings

@st.cache_resource
def ensure_vector_store() -> bool:
    """向量库不存在时自动构建。

    Streamlit 每次交互都会重跑整个脚本，cache_resource 保证
    构建逻辑每个会话只执行一次（结果跨重跑共享）。
    """
    if os.path.exists(settings.CHROMA_PERSIST_DIR) and os.listdir(settings.CHROMA_PERSIST_DIR):
        return True
    from rag import prepare_chunks, build_vector_store
    with st.spinner("首次运行：正在构建向量库（约1-2分钟）..."):
        build_vector_store(prepare_chunks())
    return True

# 先确保向量库就绪，再加载Agent（rag_search依赖检索器）
ensure_vector_store()

from agent import stream_orchestrator

# ========== 侧边栏 ==========
with st.sidebar:
    st.title("💪 健身知识问答")
    st.caption("RAG检索 + LangGraph编排 + 工具调用")

    st.divider()
    st.subheader("运行配置")
    st.text(f"模型: {settings.LLM_MODEL_NAME}")
    st.text(f"检索top_k: {settings.TOP_K}")
    st.text(f"向量库: {settings.CHROMA_COLLECTION_NAME}")

    st.divider()
    st.subheader("试试这些问题")
    EXAMPLES = [
        "深蹲的注意事项有哪些？",
        "帮我制定一份增肌训练计划",
        "帮我精确计算 (80*3)/2",
    ]
    for i, question in enumerate(EXAMPLES):
        if st.button(question, key=f"example_{i}", width="stretch"):
            # 点击后本次重跑就会走到下方的输入处理逻辑
            st.session_state.pending_question = question

    st.divider()
    if st.button("🗑️ 清空对话", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# ========== 会话状态 ==========
# messages: [{"role": "user"/"assistant", "content": 正文, "meta": 路由信息(仅assistant有)}]
if "messages" not in st.session_state:
    st.session_state.messages = []

# ========== 标题区 ==========
st.header("健身知识问答 Agent")
st.caption("简单知识问答走RAG直答（快），复杂任务走完整Agent循环（计划/计算/联网）")

# ========== 渲染历史对话 ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])

# ========== 输入处理 ==========
prompt = st.chat_input("输入你的健身问题，回车发送...") or st.session_state.pop("pending_question", None)

if prompt:
    # 先入队并渲染用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 处理并渲染助手回答（流式）
    with st.chat_message("assistant"):
        placeholder = st.empty()  # 占位符：每收到一个文本块就整体重绘
        start = time.perf_counter()
        parts = []
        meta = None
        try:
            for chunk in stream_orchestrator(prompt):
                if isinstance(chunk, dict):
                    meta = chunk
                else:
                    parts.append(chunk)
                    placeholder.markdown("".join(parts) + "▌")  # 光标符号增强"正在打字"感
            placeholder.markdown("".join(parts))  # 收尾：去掉光标符号
        except Exception as e:
            placeholder.markdown(f"出错了：{e}\n\n详细日志见 logs/run.log")
            meta = None

        if meta:
            tools = "、".join(meta["tools_used"]) if meta["tools_used"] else "无"
            meta_text = f"路由 {meta['route']} · 工具 {tools} · 耗时 {time.perf_counter() - start:.1f}s"
            st.caption(meta_text)

    if meta is None and parts:
        # 流式中断但已有部分内容：照常入历史，下次重跑仍能显示
        meta = {"route": "未知", "tools_used": []}

    st.session_state.messages.append({
        "role": "assistant",
        "content": "".join(parts),
        "meta": (
            f"路由 {meta['route']} · 工具 {'、'.join(meta['tools_used']) if meta['tools_used'] else '无'}"
            if meta else None
        ),
    })
