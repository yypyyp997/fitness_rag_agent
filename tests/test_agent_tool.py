"""Agent层单元测试：中间件纯函数、编排图结构、评估检查逻辑、计算器工具。

运行方式（项目根目录）：
    python -m unittest discover tests -v
（标准库unittest直接跑；装了pytest也可以 pytest tests/ -v）

测试对象全部是无外部依赖的纯逻辑，不调API、不碰向量库，秒级完成。
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# 把项目根加入搜索路径（tests目录没有包结构，直接相对运行）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.middleware import preprocess_input, postprocess_output, _PLAN_KEYWORDS
from agent.agent_orchestrator import (
    OrchestratorState,
    _route_by_state,
    build_orchestrator,
)
from tools.calculator import calculator

def _ai(content, tool_calls=None):
    """构造AIMessage的轻量替身：postprocess只读type/content两个属性"""
    class _M:
        pass
    m = _M()
    m.type = "ai"
    m.content = content
    if tool_calls is not None:
        m.tool_calls = tool_calls
    return m

def _tool(content):
    """构造ToolMessage的轻量替身"""
    class _M:
        pass
    m = _M()
    m.type = "tool"
    m.content = content
    return m

def _human(content):
    """构造HumanMessage的轻量替身"""
    class _M:
        pass
    m = _M()
    m.type = "human"
    m.content = content
    return m

class TestPreprocessInput(unittest.TestCase):
    """输入预处理：清洗、截断、空输入拦截"""

    def test_strips_whitespace(self):
        self.assertEqual(preprocess_input("  深蹲要领  "), "深蹲要领")

    def test_empty_returns_empty(self):
        self.assertEqual(preprocess_input(""), "")
        self.assertEqual(preprocess_input("   "), "")
        self.assertEqual(preprocess_input(None), "")

    def test_long_input_truncated_to_500(self):
        self.assertEqual(len(preprocess_input("练" * 600)), 500)

    def test_normal_input_unchanged(self):
        self.assertEqual(preprocess_input("帮我计算 (80*3)/2"), "帮我计算 (80*3)/2")

class TestPostprocessOutput(unittest.TestCase):
    """输出提取：倒序找最终AI回答（跳过工具调用轮的空content）"""

    def test_finds_last_ai_message(self):
        result = {"messages": [_human("q"), _ai("中间有工具调度的空消息", []),
                               _tool("工具结果"), _ai("最终回答")]}
        self.assertEqual(postprocess_output(result), "最终回答")

    def test_skips_empty_content(self):
        # 工具决策轮的AIMessage content为空，必须跳过
        result = {"messages": [_human("q"), _ai(""), _tool("结果"), _ai("正文")]}
        self.assertEqual(postprocess_output(result), "正文")

    def test_strips_and_returns_clean(self):
        result = {"messages": [_ai("  回答正文  \n")]}
        self.assertEqual(postprocess_output(result), "回答正文")

    def test_no_answer_returns_placeholder(self):
        self.assertEqual(postprocess_output({"messages": []}), "Agent没有生成有效回答")
        self.assertEqual(postprocess_output({}), "Agent没有生成有效回答")

    def test_only_tool_messages(self):
        result = {"messages": [_human("q"), _tool("只有工具结果")]}
        self.assertEqual(postprocess_output(result), "Agent没有生成有效回答")

class TestPlanKeywords(unittest.TestCase):
    """计划模式关键词表：防止误改/误删导致动态提示词失灵"""

    def test_core_keywords_present(self):
        for kw in ("训练计划", "帮我制定", "一周"):
            self.assertIn(kw, _PLAN_KEYWORDS)

    def test_nonempty(self):
        self.assertGreater(len(_PLAN_KEYWORDS), 0)

class TestOrchestratorStructure(unittest.TestCase):
    """编排工作流结构：状态路由函数 + 图节点/边（不执行真实节点逻辑）"""

    def test_route_by_state_reads_route_field(self):
        self.assertEqual(_route_by_state({"route": "direct"}), "direct")
        self.assertEqual(_route_by_state({"route": "agent"}), "agent")

    def test_route_by_state_defaults_to_agent(self):
        # 兜底原则：字段缺失时必须走通用路径
        self.assertEqual(_route_by_state({}), "agent")

    def test_state_schema_fields(self):
        annotations = OrchestratorState.__annotations__
        for field in ("question", "route", "answer", "tools_used"):
            self.assertIn(field, annotations)

    def test_graph_builds_with_three_nodes(self):
        graph = build_orchestrator()
        node_names = set(graph.get_graph().nodes.keys())
        for node in ("router", "direct_answer", "agent_answer"):
            self.assertIn(node, node_names)

class TestCalculator(unittest.TestCase):
    """计算器工具：白名单算术 + 危险输入拦截"""

    def test_basic_arithmetic(self):
        self.assertIn("120", calculator.invoke({"expression": "(80*3)/2"}))
        self.assertIn("2.5", calculator.invoke({"expression": "5/2"}))

    def test_math_functions(self):
        self.assertIn("2", calculator.invoke({"expression": "sqrt(4)"}))
        self.assertIn("8", calculator.invoke({"expression": "pow(2,3)"}))

    def test_division_by_zero_returns_error_hint(self):
        out = calculator.invoke({"expression": "1/0"})
        self.assertIn("计算失败", out)

    def test_syntax_error_returns_error_hint(self):
        out = calculator.invoke({"expression": "2 +* 3"})
        self.assertIn("计算失败", out)

    def test_dangerous_import_blocked(self):
        out = calculator.invoke({"expression": "__import__('os').system('echo hacked')"})
        self.assertIn("计算失败", out)

    def test_dangerous_builtin_blocked(self):
        out = calculator.invoke({"expression": "open('x')"})
        self.assertIn("计算失败", out)

class TestEvalLogic(unittest.TestCase):
    """评估模块核心检查逻辑（用桩数据，不跑真实评估）"""

    @classmethod
    def setUpClass(cls):
        from evaluation import eval_rag
        cls.eval_rag = eval_rag

    def test_check_case_all_pass(self):
        case = {"expect_route": "direct", "expect_tools": ["rag_search"],
                "expect_keywords": ["膝盖"]}
        result = {"route": "direct", "tools_used": ["rag_search"],
                  "answer": "下蹲时膝盖要对准脚尖方向"}
        check = self.eval_rag.check_case(case, result)
        self.assertTrue(check["route_ok"])
        self.assertTrue(check["tools_ok"])
        self.assertTrue(check["keywords_ok"])
        self.assertTrue(check["passed"])

    def test_check_case_route_mismatch(self):
        case = {"expect_route": "direct", "expect_tools": [], "expect_keywords": []}
        result = {"route": "agent", "tools_used": [], "answer": ""}
        check = self.eval_rag.check_case(case, result)
        self.assertFalse(check["route_ok"])
        self.assertFalse(check["passed"])

    def test_check_case_tools_subset_semantic(self):
        # 工具是包含关系：期望的都在实际列表里就算过，多的不算失败
        case = {"expect_route": "agent", "expect_tools": ["rag_search"],
                "expect_keywords": []}
        result = {"route": "agent", "tools_used": ["rag_search", "calculator"],
                  "answer": "计划"}
        check = self.eval_rag.check_case(case, result)
        self.assertTrue(check["tools_ok"])

    def test_check_case_keyword_miss(self):
        case = {"expect_route": "direct", "expect_tools": [],
                "expect_keywords": ["300", "500"]}
        result = {"route": "direct", "tools_used": [], "answer": "缺口要合理"}
        check = self.eval_rag.check_case(case, result)
        self.assertFalse(check["keywords_ok"])
        self.assertFalse(check["passed"])

    def test_questions_file_loads(self):
        cases = self.eval_rag.load_cases()
        self.assertEqual(len(cases), 8)
        # 每题结构完整
        for case in cases:
            for key in ("id", "question", "expect_route", "expect_tools", "expect_keywords"):
                self.assertIn(key, case)
            self.assertIn(case["expect_route"], ("direct", "agent"))

class TestRetryDecorator(unittest.TestCase):
    """重试装饰器：成功直达、失败重试、耗尽抛原始异常"""

    def test_success_no_retry(self):
        from utils.retry import retry_decorator
        calls = []

        @retry_decorator(max_retry=3, sleep_sec=0)
        def ok():
            calls.append(1)
            return "ok"

        self.assertEqual(ok(), "ok")
        self.assertEqual(len(calls), 1)

    def test_retry_until_success(self):
        from utils.retry import retry_decorator
        calls = []

        @retry_decorator(max_retry=3, sleep_sec=0)
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("前两次失败")
            return "ok"

        self.assertEqual(flaky(), "ok")
        self.assertEqual(len(calls), 3)

    def test_retry_exhausted_raises_original(self):
        from utils.retry import retry_decorator
        calls = []

        @retry_decorator(max_retry=2, sleep_sec=0)
        def always_fail():
            calls.append(1)
            raise KeyError("原始异常")

        with self.assertRaises(KeyError):
            always_fail()
        self.assertEqual(len(calls), 2)

class TestFileHelper(unittest.TestCase):
    """JSON文件工具：保存/读取/缺失报错（用临时目录）"""

    def test_save_and_load_json_roundtrip(self):
        from utils.file_helper import save_json, load_json
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sub", "data.json")  # 子目录不存在，顺带测ensure_dir
            save_json(path, {"name": "深蹲", "组数": 5})
            self.assertEqual(load_json(path), {"name": "深蹲", "组数": 5})

    def test_load_json_missing_raises(self):
        from utils.file_helper import load_json
        with self.assertRaises(FileNotFoundError):
            load_json(r"C:\不存在\绝不存在的文件.json")

if __name__ == "__main__":
    unittest.main(verbosity=2)
