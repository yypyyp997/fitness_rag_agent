"""失败传播路径测试：编排五层防线的完整性守护（对应 README 踩坑 9.17）。

核心断言只有一个：任何节点故障（检索崩 / 模型崩 / 落库崩 / 图执行意外失败）下，
run_orchestrator 与 stream_orchestrator 都返回结构完整的结果，绝不向调用方抛异常。

测试方法：假货注入——用 unittest.mock.patch 把检索器/模型替换成会抛异常
（或返回固定结果）的假对象，不调真实 API、不碰真实向量库，秒级完成。

隔离说明：文件顶部（任何 agent 导入之前）设置 MEMORY_DB_PATH 指向临时库，
落库相关测试写的是临时库，不碰项目真实记忆数据（同 test_memory.py 的隔离机制）。
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 测试库隔离：必须在导入 agent 之前设置（memory 路径惰性读取，但落库动作要发生在设置之后）
_TMP_DIR = tempfile.mkdtemp(prefix="fp_test_")
os.environ["MEMORY_DB_PATH"] = os.path.join(_TMP_DIR, "fp_memory.sqlite3")

from agent import agent_orchestrator as ao
from agent import memory
import agent.react_agent as ra

# tools/__init__ 的 from-import 让 tools.rag_search 属性被工具对象遮蔽，真模块从 sys.modules 拿
import tools
rag_tool_module = sys.modules["tools.rag_search"]


class _FakeRetriever:
    """正常工作的假检索器：返回一条固定文档"""
    def invoke(self, query):
        from langchain_core.documents import Document
        return [Document(page_content="下蹲时膝盖要对准脚尖方向", metadata={"source": "假库.md"})]


class _FakeLLM:
    """正常工作的假模型：invoke 返回固定内容"""
    def invoke(self, prompt):
        class _Resp:
            content = "膝盖要对准脚尖方向，循序渐进。"
        return _Resp()


class _BoomLLM:
    """会炸的假模型"""
    def invoke(self, prompt):
        raise RuntimeError("模拟模型调用崩溃")


def _boom_retriever(query=None):
    raise RuntimeError("模拟检索层崩溃")


class _BoomGraph:
    """会炸的假编排图（模拟图执行意外失败）"""
    def invoke(self, state):
        raise RuntimeError("模拟图执行意外失败")


class _FailurePathTestBase(unittest.TestCase):
    """公共 setUp/tearDown：清理模块级缓存，确保 patch 的假货被真实使用"""

    def setUp(self):
        ao._orchestrator_cache = None
        rag_tool_module._retriever_cache = None

    def tearDown(self):
        ao._orchestrator_cache = None
        rag_tool_module._retriever_cache = None


class TestRagSearchNeverRaises(_FailurePathTestBase):
    """防线1：检索工具任何故障都不抛异常，返回降级文案"""

    def test_retriever_failure_returns_hint(self):
        with patch.object(rag_tool_module, "get_retriever", _boom_retriever):
            out = rag_tool_module.rag_search.invoke({"query": "深蹲要领"})
        self.assertIsInstance(out, str)
        self.assertIn("知识库暂时不可用", out)

    def test_empty_result_returns_hint(self):
        # 检索正常但无结果：返回"没有检索到"提示（存量行为，防回归）
        class _EmptyRetriever:
            def invoke(self, query):
                return []
        with patch.object(rag_tool_module, "get_retriever", lambda: _EmptyRetriever()):
            out = rag_tool_module.rag_search.invoke({"query": "冷门问题"})
        self.assertIn("没有检索到", out)


class TestDirectPathNeverRaises(_FailurePathTestBase):
    """防线2：direct 支路检索/生成失败都降级，经 run_orchestrator 全链验证不炸"""

    def _run_direct(self):
        ao._orchestrator_cache = None
        with patch.object(ao, "route_question", lambda s: {"route": "direct"}):
            return ao.run_orchestrator("深蹲的注意事项有哪些？")

    def test_retrieval_failure_degraded(self):
        with patch.object(rag_tool_module, "get_retriever", _boom_retriever):
            result = self._run_direct()
        self.assertEqual(set(result.keys()), {"answer", "route", "tools_used"})
        self.assertIn("知识库暂时不可用", result["answer"])
        self.assertEqual(result["tools_used"], ["rag_search"])

    def test_generation_failure_degraded(self):
        # 检索正常、生成崩：降级文案必须区分于检索故障
        with patch.object(rag_tool_module, "get_retriever", lambda: _FakeRetriever()), \
             patch.object(ao, "get_llm", lambda: _BoomLLM()):
            result = self._run_direct()
        self.assertIn("模型服务暂时不可用", result["answer"])
        self.assertEqual(result["tools_used"], ["rag_search"])

    def test_success_path_normal_answer(self):
        # 快乐路径回归：假检索器+假模型正常时，防线不误伤、回答正常产出
        with patch.object(rag_tool_module, "get_retriever", lambda: _FakeRetriever()), \
             patch.object(ao, "get_llm", lambda: _FakeLLM()):
            result = self._run_direct()
        self.assertEqual(result["route"], "direct")
        self.assertIn("膝盖", result["answer"])
        self.assertNotIn("暂时不可用", result["answer"])


class TestAgentPathNeverRaises(_FailurePathTestBase):
    """防线3：agent 支路模型构建/调用失败降级，节点不抛异常"""

    def test_model_failure_degraded(self):
        # react_agent 模块里绑定的 get_llm 抛异常 → agent_answer 防线3接住
        with patch.object(ao, "route_question", lambda s: {"route": "agent"}), \
             patch.object(ra, "get_llm", side_effect=RuntimeError("模拟模型构建失败")):
            result = ao.run_orchestrator("帮我计算 (80*3)/2")
        self.assertEqual(set(result.keys()), {"answer", "route", "tools_used"})
        self.assertIn("模型服务暂时不可用", result["answer"])


class TestEntryBottomLine(_FailurePathTestBase):
    """防线4/5：历史读写失败不炸、落库失败不炸、降级不落库、图执行意外失败有总兜底"""

    def _happy_direct(self, session_id="fp_entry"):
        """构造一条正常完成的 direct 链路（全假货，零 API）"""
        ao._orchestrator_cache = None
        with patch.object(rag_tool_module, "get_retriever", lambda: _FakeRetriever()), \
             patch.object(ao, "get_llm", lambda: _FakeLLM()), \
             patch.object(ao, "route_question", lambda s: {"route": "direct"}):
            return ao.run_orchestrator("深蹲要领", session_id=session_id)

    def test_history_read_failure_no_raise(self):
        # 读取历史失败：按无历史继续，回答照常产出（独立会话ID，避免落库计数互串）
        with patch.object(memory, "get_recent_messages", side_effect=RuntimeError("记忆库锁死")):
            result = self._happy_direct(session_id="fp_histread")
        self.assertIn("膝盖", result["answer"])

    def test_save_history_failure_no_raise(self):
        # 落库失败：记日志不上抛，回答正常返回（独立会话ID）
        with patch.object(memory, "append_message", side_effect=RuntimeError("磁盘写保护")):
            result = self._happy_direct(session_id="fp_histsave")
        self.assertIn("膝盖", result["answer"])

    def test_degraded_answer_not_saved(self):
        # 降级回答不写入多轮记忆（避免故障对话污染历史）
        ao._orchestrator_cache = None
        with patch.object(rag_tool_module, "get_retriever", _boom_retriever), \
             patch.object(ao, "route_question", lambda s: {"route": "direct"}):
            result = ao.run_orchestrator("深蹲要领", session_id="fp_degraded")
        self.assertIn("知识库暂时不可用", result["answer"])
        self.assertEqual(memory.get_recent_messages("fp_degraded"), [])

    def test_normal_answer_saved(self):
        # 对照组：正常回答要落库（user+assistant 两条）
        self._happy_direct()
        history = memory.get_recent_messages("fp_entry")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")

    def test_graph_unexpected_failure_degraded_dict(self):
        # 防线5：图执行意外失败 → 返回降级 dict，绝不 raise
        with patch.object(ao, "build_orchestrator", lambda: _BoomGraph()):
            result = ao.run_orchestrator("深蹲要领")
        self.assertEqual(set(result.keys()), {"answer", "route", "tools_used"})
        self.assertIn("模型服务暂时不可用", result["answer"])

    def test_stream_direct_failure_still_yields_meta(self):
        # 流式 direct：检索崩 → 仍有文本块 + 收尾 meta dict，生成器不抛异常
        with patch.object(rag_tool_module, "get_retriever", _boom_retriever), \
             patch.object(ao, "route_question", lambda s: {"route": "direct"}):
            chunks = list(ao.stream_orchestrator("深蹲要领"))
        self.assertTrue(len(chunks) >= 1)
        meta = chunks[-1]
        self.assertIsInstance(meta, dict)
        self.assertIn("answer", meta)
        self.assertTrue(meta["answer"])
        self.assertIn(meta["answer"], ao._DEGRADED_ANSWERS)

    def test_stream_agent_failure_still_yields_meta(self):
        # 流式 agent：run_orchestrator 意外抛（理论上防线5已兜住）→ 流式层再兜一次
        with patch.object(ao, "run_orchestrator", side_effect=RuntimeError("意外失败")), \
             patch.object(ao, "route_question", lambda s: {"route": "agent"}):
            chunks = list(ao.stream_orchestrator("帮我制定计划"))
        self.assertTrue(len(chunks) >= 1)
        meta = chunks[-1]
        self.assertIsInstance(meta, dict)
        self.assertIn("模型服务暂时不可用", meta["answer"])

    def test_degraded_answers_registry_complete(self):
        # 契约：降级文案登记表覆盖全部降级文案（新增降级文案必须登记，否则会误落库）
        self.assertIn(ao._DEGRADED_RETRIEVAL, ao._DEGRADED_ANSWERS)
        self.assertIn(ao._DEGRADED_GENERATION, ao._DEGRADED_ANSWERS)
        self.assertTrue(ao._is_degraded(ao._DEGRADED_RETRIEVAL))
        self.assertFalse(ao._is_degraded("正常回答内容"))
        self.assertFalse(ao._is_degraded(""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
