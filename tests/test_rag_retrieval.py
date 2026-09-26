"""RAG链路单元测试：文档加载器参数、切分参数边界、提示词加载与占位符。

运行方式（项目根目录）：
    python -m unittest discover tests -v

说明：不调Embedding接口、不碰chroma_db，只验证纯逻辑与配置正确性。
（向量召回质量属于集成测试范畴，由 evaluation/eval_rag.py 的真实评估覆盖）
"""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

class TestRagConfig(unittest.TestCase):
    """RAG配置参数：边界值合法性（改配置时第一时间发现低级错误）"""

    def test_chunk_params_positive(self):
        self.assertGreater(settings.CHUNK_SIZE, 0)
        self.assertGreater(settings.CHUNK_OVERLAP, 0)

    def test_chunk_overlap_less_than_size(self):
        # overlap >= size 会导致RecursiveCharacterTextSplitter行为异常
        self.assertLess(settings.CHUNK_OVERLAP, settings.CHUNK_SIZE)

    def test_top_k_positive(self):
        self.assertGreater(settings.TOP_K, 0)

    def test_data_paths_defined(self):
        for attr in ("RAW_DATA_PATH", "PROCESSED_DATA_PATH", "CHROMA_PERSIST_DIR"):
            self.assertTrue(getattr(settings, attr))

    def test_raw_data_exists(self):
        import os
        self.assertTrue(
            os.listdir(os.path.join(PROJECT_ROOT, settings.RAW_DATA_PATH)),
            "data/raw 目录为空：请放入知识库文档",
        )

class TestSplitter(unittest.TestCase):
    """文档切分：用内存文档验证切分行为（不依赖磁盘文件）"""

    def test_split_long_document(self):
        from rag.splitter import split_documents
        from langchain_core.documents import Document

        doc = Document(page_content="深蹲要点。" * 100)  # 500字
        chunks = split_documents([doc])
        self.assertGreater(len(chunks), 1)
        # 每个chunk不超过chunk_size（含overlap的容差）
        for c in chunks:
            self.assertLessEqual(len(c.page_content), settings.CHUNK_SIZE)

    def test_short_document_single_chunk(self):
        from rag.splitter import split_documents
        from langchain_core.documents import Document

        doc = Document(page_content="很短的文档")
        chunks = split_documents([doc])
        self.assertEqual(len(chunks), 1)

    def test_metadata_preserved(self):
        from rag.splitter import split_documents
        from langchain_core.documents import Document

        doc = Document(page_content="内容" * 100, metadata={"source": "测试.md"})
        chunks = split_documents([doc])
        for c in chunks:
            self.assertEqual(c.metadata.get("source"), "测试.md")

class TestPromptLoading(unittest.TestCase):
    """提示词模块：三个提示词可加载、rag_prompt占位符齐全"""

    def test_all_prompts_loadable(self):
        from prompts import get_prompt
        for name in ("rag_prompt", "agent_system", "tool_desc"):
            content = get_prompt(name)
            self.assertTrue(content, f"{name} 加载结果为空")

    def test_rag_prompt_placeholders(self):
        # rag_prompt必须有这两个占位符，编排层direct支路靠它们format
        from prompts import get_prompt
        content = get_prompt("rag_prompt")
        self.assertIn("{context}", content)
        self.assertIn("{question}", content)

    def test_generation_params_has_temperature(self):
        from prompts import get_generation_params
        params = get_generation_params()
        self.assertIn("temperature", params)
        self.assertIsInstance(params["temperature"], (int, float))

class TestPromptCommentStripping(unittest.TestCase):
    """提示词加载的注释剔除逻辑：#开头的行不该出现在最终提示词里"""

    def test_comment_lines_stripped(self):
        # 直接构造一个带注释的提示词文件场景：检查已加载的正式提示词里
        # 不应残留明显的注释行（以"# "开头且不是正文标题的行）
        # 注：agent_system.md的正文以markdown标题组织，标题是"#"开头——
        # 因此这里只验证rag_prompt（设计上正文无#标题，注释已在加载时剔除）
        from prompts import get_prompt
        content = get_prompt("rag_prompt")
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                self.fail(f"rag_prompt里残留注释行: {stripped}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
