from pathlib import Path

from langchain_core.tools import tool

from rag.retrieval import get_retriever
from utils import get_logger

logger = get_logger(__name__)

_retriever_cache = None

def _get_cached_retriever():
    """检索器全局只建一次，Agent 多轮调用时不重复初始化"""
    global _retriever_cache
    if _retriever_cache is None:
        _retriever_cache = get_retriever()
    return _retriever_cache

@tool
def rag_search(query: str) -> str:
    """检索本地健身知识库。输入自然语言问题，返回知识库中最相关的知识片段。回答健身专业问题（动作要领、训练原理、计划安排）时优先使用本工具。"""
    logger.info(f"rag_search 检索: {query}")
    docs = _get_cached_retriever().invoke(query)
    if not docs:
        return "知识库中没有检索到相关内容。"

    # 拼接检索结果并带上来源，方便Agent在回答里注明出处
    parts = []
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "未知来源")).name
        parts.append(f"【片段{i}｜来源: {source}】\n{doc.page_content}")
    return "\n\n".join(parts)
