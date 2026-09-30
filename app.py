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
from agent import memory

def _session_id() -> str:
    """当前活动会话ID：无则新建（UI会话管理的核心）"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
        _new_session()
    return st.session_state.session_id

def _new_session() -> None:
    """新建会话：立即在记忆库登记（空会话也出现在侧边栏列表），界面清空"""
    import uuid
    new_id = f"st-{uuid.uuid4().hex[:12]}"
    memory.create_session(new_id, title="新会话")
    st.session_state.session_id = new_id
    st.session_state.messages = []
    st.session_state.pop("named_session", None)  # 新会话允许重新自动命名

def _switch_session(target_id: str) -> None:
    """切换会话：从记忆库回放最近历史到界面"""
    history = memory.get_recent_messages(target_id, limit=50)  # 回放最近25轮
    st.session_state.session_id = target_id
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"]} for m in history
    ]

def _auto_title_if_first(question: str) -> None:
    """首问自动命名：会话第一条提问的前20个字符作为会话标题

    让侧边栏列表显示"我身高175体重80，我的bmi是多少"而不是一串随机ID。
    """
    if st.session_state.get("named_session"):
        return  # 本会话已命名过
    if memory.get_session_title(_session_id(), default="新会话") == "新会话":
        title = question.strip().replace("\n", " ")[:20]
        if title:
            memory.rename_session(_session_id(), title)
    st.session_state.named_session = True

# ========== 侧边栏 ==========
with st.sidebar:
    st.title("💪 健身知识问答")
    st.caption("RAG检索 + LangGraph编排 + 工具调用")

    st.divider()
    st.subheader("会话历史")
    if st.button("＋ 新建会话", width="stretch", type="primary"):
        _new_session()
        st.rerun()

    sessions = memory.list_sessions()
    if sessions:
        st.caption(f"共 {len(sessions)} 个会话（点击切换）")
        for sess in sessions:
            sid = sess["session_id"]
            is_current = (sid == st.session_state.get("session_id"))
            title = sess.get("title") or ("★ 当前" if is_current else sid[:10] + "…")
            row1, row2 = st.columns([0.82, 0.18], vertical_alignment="center")
            with row1:
                label = ("▶ " if is_current else "") + title
                if st.button(label, key=f"sw_{sid}", width="stretch",
                             help=f"{sess['msg_count']} 条消息 · 最近活跃 {sess['last_active']}"):
                    if not is_current:
                        _switch_session(sid)
                        st.rerun()
            with row2:
                if st.button("🗑", key=f"del_{sid}", help="删除该会话"):
                    memory.delete_session(sid)
                    if is_current:
                        _new_session()
                    st.rerun()
    else:
        st.caption("还没有会话，点上方按钮新建")

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
    st.subheader("多轮记忆")
    use_memory = st.toggle("启用多轮对话", value=True,
                           help="开启后同一会话内可追问（如：那硬拉呢？）")
    if st.button("清空本会话记忆", width="stretch", disabled=not use_memory):
        n = memory.clear_session(_session_id())
        st.session_state.messages = []
        st.session_state.named_session = False  # 清空后允许重新自动命名
        st.toast(f"已清空{n}条记忆")
        st.rerun()

    st.divider()
    if st.button("🗑️ 清空对话", width="stretch"):
        st.session_state.messages = []
        st.rerun()

# ========== 会话状态 ==========
# messages: [{"role": "user"/"assistant", "content": 正文, "meta": 路由信息(仅assistant有)}]
if "messages" not in st.session_state:
    st.session_state.messages = []
# 启动时确保有一个已登记的活动会话（首次打开页面即建会话并入库）
_session_id()

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
    # 首问自动命名会话（放在最前，重跑时列表标题即时更新）
    if use_memory:
        _auto_title_if_first(prompt)

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
            sid = _session_id() if use_memory else None
            for chunk in stream_orchestrator(prompt, session_id=sid):
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
