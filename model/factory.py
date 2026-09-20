from langchain_dashscope import ChatDashScope, DashScopeEmbeddings
from config import settings
from utils import get_logger, retry_decorator

logger = get_logger(__name__)


def get_llm():
    """获取大模型实例：通义千问"""
    logger.info(f"初始化LLM模型: {settings.LLM_MODEL_NAME}")
    llm = ChatDashScope(
        model_name=settings.LLM_MODEL_NAME,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
        temperature=0.1
    )
    return llm


@retry_decorator(max_retry=3)
def get_embedding():
    """获取Embedding向量模型实例：TextEmbedding V4"""
    logger.info(f"初始化Embedding模型: {settings.EMBEDDING_MODEL_NAME}")
    embedding = DashScopeEmbeddings(
        model=settings.EMBEDDING_MODEL_NAME,
        dashscope_api_key=settings.DASHSCOPE_API_KEY
    )
    return embedding
