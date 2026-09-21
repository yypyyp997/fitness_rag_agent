from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from config import settings
from prompts import get_generation_params
from utils import get_logger, retry_decorator

logger = get_logger(__name__)

def get_llm():
    """获取大模型实例：通义千问（生成参数从 prompt_config.yaml 读取）"""
    logger.info(f"初始化LLM模型: {settings.LLM_MODEL_NAME}")
    params = get_generation_params()
    llm = ChatTongyi(
        model_name=settings.LLM_MODEL_NAME,
        dashscope_api_key=settings.DASHSCOPE_API_KEY,
        model_kwargs=params  # temperature等参数必须走model_kwargs才会传给DashScope接口
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
