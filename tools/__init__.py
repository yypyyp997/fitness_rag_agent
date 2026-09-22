from .rag_search import rag_search
from .calculator import calculator
from .search_web import search_web

def get_tools():
    """返回Agent可用的全部工具列表（提交点6编排时直接使用）"""
    return [rag_search, calculator, search_web]

__all__ = ["rag_search", "calculator", "search_web", "get_tools"]
