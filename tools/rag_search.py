"""检索工具：向量知识库查询（RAG直答支路与Agent支路共用）。

故障兜底（防线1，对应 README 9.17）：检索器初始化失败与检索执行失败
都在工具内部捕获并降级为自然语言提示，绝不向调用方抛异常。

为什么在工具层兜底（而不是只靠编排层节点保护）：
- Agent支路里 ToolNode 的默认策略只兜参数校验错，普通异常会穿透炸穿子图
  （探针实证：During task with name 'tools'）——工具自身捕获是唯一
  能让「工具失败也变成有效输出」的位置；
- 返回的降级文案会作为工具结果喂回模型，模型能向用户自然解释「知识库
  暂时不可用」，与 README 9.7 的设计哲学一致：工具的"失败"也是有效输出。
"""
from pathlib import Path

from langchain_core.tools import tool

from rag.retrieval import get_retriever
from utils import get_logger

logger = get_logger(__name__)

_retriever_cache = None

# 检索层降级文案（模块级常量）：编排层靠它识别「检索已失败」并短路生成（省一次必然无意义的 LLM 调用）
RETRIEVAL_DEGRADED_HINT = "知识库暂时不可用（检索服务异常），无法基于本地知识库回答该问题。请稍后重试，或检查向量库是否已构建。"

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
    try:
        docs = _get_cached_retriever().invoke(query)
    except Exception as e:
        # 防线1：检索层任何异常（向量库缺失/损坏、Embedding故障、网络问题）
        # 都降级为自然语言提示。Agent支路里这段文案会喂回模型自行解释；
        # direct支路里编排层会识别该文案并短路生成（不浪费模型调用）。
        logger.error(f"rag_search 检索失败，返回降级提示: {e}")
        return RETRIEVAL_DEGRADED_HINT

    if not docs:
        return "知识库中没有检索到相关内容。"

    # 拼接检索结果并带上来源，方便Agent在回答里注明出处
    parts = []
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "未知来源")).name
        parts.append(f"【片段{i}｜来源: {source}】\n{doc.page_content}")
    return "\n\n".join(parts)
