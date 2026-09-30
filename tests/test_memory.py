"""多轮记忆单元测试：独立临时库，不碰项目真实记忆数据。

隔离机制：MEMORY_DB_PATH 在任何 agent 导入前设置；memory 模块的
库路径为惰性读取（函数内读环境变量），配合下方回归测试硬断言。
"""
import sys
import os
import unittest
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_TMP_DIR = tempfile.mkdtemp(prefix="mem_test_")
os.environ["MEMORY_DB_PATH"] = os.path.join(_TMP_DIR, "test_memory.sqlite3")

from agent import memory

class TestMemory(unittest.TestCase):
    """SQLite记忆存取：追加/窗口/清空"""

    def test_memory_isolated_from_real_db(self):
        """回归测试：环境变量必须生效，确保写的是临时库不是项目的真实库。

        曾踩的坑：库路径用模块级常量在导入时锁定，测试按字母序先跑的
        test_agent_tool 连带导入 agent 包，等本文件设环境变量时已晚——
        两例测试失败且把测试数据写进了真实库。修复后路径惰性读取，
        这里硬断言路径必须跟随环境变量。
        """
        self.assertEqual(memory._db_path(), os.environ["MEMORY_DB_PATH"],
                         "memory库路径未跟随MEMORY_DB_PATH，测试隔离失效！")
        self.assertNotEqual(memory._db_path(), "data/memory.sqlite3")

    def test_append_and_get_recent(self):
        memory.append_message("s1", "user", "增肌吃多少蛋白？")
        memory.append_message("s1", "assistant", "普通训练者1.6~2.0g/kg", route="direct", tools_used=["rag_search"])
        history = memory.get_recent_messages("s1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["content"], "普通训练者1.6~2.0g/kg")

    def test_window_limit_keeps_latest(self):
        for i in range(8):
            memory.append_message("s2", "user", f"问题{i}")
        history = memory.get_recent_messages("s2", limit=6)
        self.assertEqual(len(history), 6)
        # 窗口保留的是最新的6条：问题2~问题7
        self.assertEqual(history[0]["content"], "问题2")
        self.assertEqual(history[-1]["content"], "问题7")

    def test_session_isolation(self):
        memory.append_message("s3", "user", "A会话的问题")
        memory.append_message("s4", "user", "B会话的问题")
        h3 = memory.get_recent_messages("s3")
        h4 = memory.get_recent_messages("s4")
        self.assertEqual(len(h3), 1)
        self.assertEqual(len(h4), 1)
        self.assertEqual(h3[0]["content"], "A会话的问题")

    def test_clear_session(self):
        memory.append_message("s5", "user", "问题1")
        memory.append_message("s5", "assistant", "回答1")
        deleted = memory.clear_session("s5")
        self.assertEqual(deleted, 2)
        self.assertEqual(memory.get_recent_messages("s5"), [])

    def test_format_history_block(self):
        history = [
            {"role": "user", "content": "增肌吃什么"},
            {"role": "assistant", "content": "蛋白质与碳水"},
        ]
        block = memory.format_history_block(history)
        self.assertIn("对话历史", block)
        self.assertIn("增肌吃什么", block)
        # 空历史返回空串（不注入）
        self.assertEqual(memory.format_history_block([]), "")

if __name__ == "__main__":
    unittest.main(verbosity=2)
