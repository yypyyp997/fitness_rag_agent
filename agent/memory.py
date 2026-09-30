"""多轮对话记忆：SQLite 持久化 + 滑动窗口注入。

设计决策（面试可讲）：
- 为什么不用 LangChain 自带的 Memory 类：我们要"存储"与"注入"解耦——
  存储层按session全量落库（可审计、可恢复），注入层只在调用编排层时
  按需取最近N条（上下文经济性）。框架Memory类把两件事耦在一起，反而不好控制。
- 为什么用 SQLite：单文件零运维、标准库自带、线程安全（check_same_thread=False
  + 每次操作独立连接），对单机服务完全够用；换 Postgres 只需改这一个文件。
- 为什么窗口取6条：约3轮对话。实验上更长的窗口对质量提升有限，却线性增加
  每次请求的token成本；摘要/向量召回是更优的方案，留作迭代。
- 踩坑记录：库路径不能用模块级常量在导入时锁定——unittest按字母序先跑的
  其它测试会连带导入本模块，导致后设的MEMORY_DB_PATH失效，测试数据写进
  真实库（曾致2例测试失败+真实库污染）。路径必须在函数内惰性读取。
"""
import os
import sqlite3
import threading
from typing import List, Optional

from utils import get_logger

logger = get_logger(__name__)

def _db_path() -> str:
    """库路径惰性读取：每次调用时读环境变量（测试隔离依赖这一点）"""
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
            conn.commit()
        finally:
            conn.close()
        _initialized = True
        logger.info(f"对话记忆库就绪: {_db_path()}")

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

def clear_session(session_id: str) -> int:
    """清空指定会话，返回删除的条数"""
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
