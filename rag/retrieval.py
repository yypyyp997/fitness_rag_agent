from langchain_core.retrievers import BaseRetriever
from config import settings
from utils import get_logger
from rag.vector_store import get_vector_store

logger = get_logger(__name__)


def get_retriever() -> BaseRetriever:
    """获取向量检索器，top_k从配置读取"""
    vector_db = get_vector_store()
    retriever = vector_db.as_retriever(
        search_kwargs={"k": settings.TOP_K}
    )
    logger.info(f"检索器初始化完成, top_k={settings.TOP_K}")
    return retriever
