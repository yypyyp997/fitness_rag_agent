import os

from langchain_core.tools import tool
from utils import get_logger

logger = get_logger(__name__)

def _pick(obj, key, default=""):
    """兼容不同版本tavily SDK的返回结构（dict或对象）"""
    if isinstance(obj, dict):
        return obj.get(key, default) or default
    return getattr(obj, key, default) or default

@tool
def search_web(query: str) -> str:
    """联网搜索最新资讯。输入搜索关键词，返回网页搜索结果摘要。仅当本地知识库没有相关内容时使用。"""
    try:
        from tavily import TavilyClient
    except ImportError:
        return "联网搜索不可用：未安装 tavily-python（pip install tavily-python）"

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "联网搜索未配置：请在项目根目录 .env 中设置 TAVILY_API_KEY（tavily.com 免费注册），配置后重启生效。"

    try:
        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, max_results=3)
        results = _pick(resp, "results", [])
        if not results:
            return "联网搜索没有返回结果。"

        parts = []
        for i, item in enumerate(results, 1):
            title = _pick(item, "title")
            url = _pick(item, "url")
            content = str(_pick(item, "content"))[:400]
            parts.append(f"【结果{i}】{title}\n链接: {url}\n摘要: {content}")
        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"联网搜索失败: {str(e)}")
        return f"联网搜索失败: {str(e)}。请检查 TAVILY_API_KEY 与网络后重试。"
