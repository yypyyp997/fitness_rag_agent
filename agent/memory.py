# agent/memory.py
"""多轮对话记忆：SQLite 持久化 + 滑动窗口注入。

设计决策：
- 为什么不用 LangChain 自带的 Memory 类：我们要"存储"与"注入"解耦——
  存储层按session全量落库（可审计、可恢复），注入层只在调用编排层时
  按需取最近N条（上下文经济性）。框架Memory类把两件事耦在一起，反而不好控制。
- 为什么用 SQLite：单文件零运维、标准库自带、线程安全（check_same_thread=False
  + 每次操作独立连接），对单机服务完全够用；换 Postgres 只需改这一个文件。
- 为什么窗口取6条：约3轮对话。实验上更长的窗口对质量提升有限，却线性增加
  每次请求的token成本；摘要/向量召回是更优的方案，留作迭代。
- 会话生命周期：create_session 在"新建会话"点击时立即登记（ChatGPT式体验，
  空会话也在列表可见）；标题默认"新会话"，首条提问后自动改为提问前20字；
  delete_session 连带清元信息（避免删掉的会话残留列表）。
"""
import os
import sqlite3
import threading
from typing import List, Optional

from utils import get_logger

logger = get_logger(__name__)

# 库路径【惰性读取】：必须在函数里读环境变量，不能用模块级常量——
# 否则同进程里先导入本模块的代码（如测试按字母序先跑其它用例）会把路径
# 锁定在默认值，后设的 MEMORY_DB_PATH 失效，测试就会写进真实库（已踩坑）
def _db_path() -> str:
    return os.getenv("MEMORY_DB_PATH", "data/memory.sqlite3")
_INIT_LOCK = threading.Lock()
_initialized = False

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,             -- 'user' / 'assistant'
    content TEXT NOT NULL,
    route TEXT,                     -- assistant消息附带的元信息
    tools_used TEXT,                -- JSON数组字符串
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""
_SESSION_META_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_meta (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# 会话列表查询：UNION 同时覆盖两类来源——
#   1) session_meta 里已登记但还没有消息的新会话（点击新建后立即可见）
#   2) messages 里有对话记录的会话（含旧格式：没有元信息行的历史会话）
# 两边都存在的会话合并计数，标题取元信息表（消息侧贡献空串，MAX自然选中非空标题）
_SESSIONS_SQL = """
SELECT sid AS session_id,
       SUM(cnt) AS msg_count,
       MAX(last_active) AS last_active,
       MAX(title) AS title
FROM (
    SELECT sm.session_id AS sid, 0 AS cnt, sm.created_at AS last_active, sm.title AS title
    FROM session_meta sm
    UNION ALL
    SELECT msg.session_id AS sid, COUNT(*) AS cnt, MAX(msg.created_at) AS last_active, '' AS title
    FROM messages msg
    GROUP BY msg.session_id
)
GROUP BY sid
ORDER BY last_active DESC
"""

def _get_conn() -> sqlite3.Connection:
    """每次操作独立连接（避免跨线程共享连接的坑），用完即关"""
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")  # 读写并发友好
    return conn

def _ensure_init() -> None:
    """建表（进程内只做一次，锁防止并发首调竞争）"""
    global _initialized
    if _initialized:
        return
    with _INIT_LOCK:
        if _initialized:
            return
        os.makedirs(os.path.dirname(_db_path()) or ".", exist_ok=True)
        conn = _get_conn()
        try:
            conn.executescript(_SCHEMA)
            conn.executescript(_SESSION_META_SCHEMA)
            conn.commit()
        finally:
            conn.close()
        _initialized = True
        logger.info(f"对话记忆库就绪: {_db_path()}")

def create_session(session_id: str, title: str = "新会话") -> None:
    """登记一个新会话（点击"新建会话"时立即调用，空会话也出现在列表里）

    ON CONFLICT DO NOTHING：同一ID重复登记（如界面重跑）静默跳过。
    """
    _ensure_init()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO session_meta (session_id, title) VALUES (?, ?) "
            "ON CONFLICT(session_id) DO NOTHING",
            (session_id, title[:50]),
        )
        conn.commit()
    finally:
        conn.close()

def append_message(
    session_id: str,
    role: str,
    content: str,
    route: Optional[str] = None,
    tools_used: Optional[list] = None,
) -> None:
    """追加一条对话记录（回答成功生成后才调用，失败不污染历史）"""
    _ensure_init()
    import json
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, route, tools_used) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, route, json.dumps(tools_used or [], ensure_ascii=False)),
        )
        conn.commit()
    finally:
        conn.close()

def get_recent_messages(session_id: str, limit: int = 6) -> List[dict]:
    """取最近N条（时间正序返回），滑动窗口"""
    _ensure_init()
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "SELECT role, content FROM ("
            "  SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?"
            ") ORDER BY id ASC",
            (session_id, limit),
        )
        return [{"role": r[0], "content": r[1]} for r in cursor.fetchall()]
    finally:
        conn.close()

def list_sessions() -> List[dict]:
    """列出全部会话（按最近活跃倒序）：session_id / 标题 / 消息数 / 最后活跃"""
    _ensure_init()
    conn = _get_conn()
    try:
        cursor = conn.execute(_SESSIONS_SQL)
        return [
            {
                "session_id": r[0],
                "msg_count": r[1],
                "last_active": r[2],
                "title": r[3] or "",
            }
            for r in cursor.fetchall()
        ]
    finally:
        conn.close()

def rename_session(session_id: str, title: str) -> None:
    """重命名会话（写会话元信息表）

    标题存独立表而不是改消息行：标题是会话级属性，与消息解耦。
    """
    _ensure_init()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO session_meta (session_id, title) VALUES (?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET title = excluded.title",
            (session_id, title[:50]),
        )
        conn.commit()
    finally:
        conn.close()

def get_session_title(session_id: str, default: str = "") -> str:
    """读会话标题；无记录返回 default"""
    _ensure_init()
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT title FROM session_meta WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row[0] if row else default
    finally:
        conn.close()

def clear_session(session_id: str) -> int:
    """清空指定会话的对话记录（保留会话本身在列表里），返回删除的条数"""
    _ensure_init()
    conn = _get_conn()
    try:
        cursor = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
        deleted = cursor.rowcount
        logger.info(f"会话 {session_id} 已清空，删除 {deleted} 条")
        return deleted
    finally:
        conn.close()

def delete_session(session_id: str) -> int:
    """彻底删除会话：对话记录 + 元信息一起删（列表里不再出现），返回删除的消息条数"""
    _ensure_init()
    conn = _get_conn()
    try:
        cursor = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        deleted = cursor.rowcount
        conn.execute("DELETE FROM session_meta WHERE session_id = ?", (session_id,))
        conn.commit()
        logger.info(f"会话 {session_id} 已删除，清掉 {deleted} 条消息")
        return deleted
    finally:
        conn.close()

def format_history_block(history: List[dict]) -> str:
    """把历史消息格式化为提示词块（direct支路用，拼在检索模板前）"""
    if not history:
        return ""
    lines = ["【对话历史（供理解当前问题的上下文，不要重复回答历史问题）】"]
    for m in history:
        prefix = "用户" if m["role"] == "user" else "助手"
        content = m["content"][:200]  # 单条截断，防止历史长回答撑爆上下文
        lines.append(f"{prefix}: {content}")
    lines.append("【历史结束，以下是当前问题的回答】\n")
    return "\n".join(lines)
