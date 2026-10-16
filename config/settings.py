import os
from dotenv import load_dotenv

# 加载根目录下的.env环境变量（存放API密钥）
load_dotenv()

# 项目根目录（config/ 的上一级）：所有路径锚定到这里，与运行时工作目录无关
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Settings:
    # ========== DashScope 大模型与Embedding配置 ==========
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    LLM_MODEL_NAME: str = "qwen-max"
    EMBEDDING_MODEL_NAME: str = "text-embedding-v4"

    # ========== 向量库 Chroma 配置 ==========
    CHROMA_PERSIST_DIR: str = os.path.join(BASE_DIR, "chroma_db")
    CHROMA_COLLECTION_NAME: str = "fitness_knowledge"

    # ========== RAG 检索参数（消融实验就在这里改） ==========
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 30
    TOP_K: int = 3

    # ========== 文件路径配置 ==========
    RAW_DATA_PATH: str = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DATA_PATH: str = os.path.join(BASE_DIR, "data", "processed")
    LOG_DIR: str = os.path.join(BASE_DIR, "logs")

    # ========== Rerank重排序配置 ==========
    RERANK_MODEL_NAME: str = "BAAI/bge-reranker-base"

# 实例化全局单例配置，项目其他地方直接 from config import settings
settings = Settings()
