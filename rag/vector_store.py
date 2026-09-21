import pickle
import hashlib
from pathlib import Path
from langchain_chroma import Chroma
from model import get_embedding
from config import settings
from utils import get_logger

logger = get_logger(__name__)

# processed缓存文件路径
CACHE_FILE = Path(settings.PROCESSED_DATA_PATH) / "chunks_cache.pkl"
MD5_RECORD = Path(settings.PROCESSED_DATA_PATH) / "raw_md5.txt"
RAW_DATA_DIR = Path(settings.RAW_DATA_PATH)


def calc_dir_md5(dir_path: Path) -> str:
    """计算raw目录全部文件md5，用来检测原始知识库是否变更"""
    md5 = hashlib.md5()
    for file in sorted(dir_path.glob("*")):
        if file.is_file():
            with open(file, "rb") as f:
                while chunk := f.read(4096):
                    md5.update(chunk)
    return md5.hexdigest()


def save_chunks_cache(docs):
    """把切分后的Document对象保存到processed缓存"""
    # 确保processed文件夹存在
    Path(settings.PROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(docs, f)
    current_md5 = calc_dir_md5(RAW_DATA_DIR)
    with open(MD5_RECORD, "w", encoding="utf-8") as f:
        f.write(current_md5)
    logger.info("✅ 切分后的文档已写入processed缓存")


def load_chunks_cache():
    """加载缓存，校验MD5，文件改动则缓存失效返回None"""
    if not CACHE_FILE.exists() or not MD5_RECORD.exists():
        logger.info("ℹ️ 不存在chunk缓存")
        return None

    with open(MD5_RECORD, "r", encoding="utf-8") as f:
        old_md5 = f.read().strip()
    current_md5 = calc_dir_md5(RAW_DATA_DIR)

    if old_md5 != current_md5:
        logger.info("🔔 raw知识库文件发生变更，缓存失效，需要重新切分文档")
        return None

    with open(CACHE_FILE, "rb") as f:
        cached_docs = pickle.load(f)
    logger.info("✅ 成功从processed加载缓存chunks，跳过文档加载+文本切分")
    return cached_docs


def prepare_chunks():
    """顶层入口：获取切分文档，优先读缓存；缓存失效则加载原始文档+切分"""
    from rag.loader import load_documents
    from rag.splitter import split_documents

    cached = load_chunks_cache()
    if cached is not None:
        return cached

    logger.info("🔄 无有效缓存，重新加载原始文档并切分")
    raw_docs = load_documents(RAW_DATA_DIR)
    split_docs = split_documents(raw_docs)
    save_chunks_cache(split_docs)
    return split_docs


def build_vector_store(split_docs):
    """构建向量库并持久化保存到本地"""
    embedding = get_embedding()
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.CHROMA_COLLECTION_NAME
    )
    logger.info("向量库构建完成，已持久化")
    return vector_db


def get_vector_store():
    """读取本地已经建好的向量库，避免重复向量化"""
    embedding = get_embedding()
    vector_db = Chroma(
        embedding_function=embedding,
        persist_directory=settings.CHROMA_PERSIST_DIR,
        collection_name=settings.CHROMA_COLLECTION_NAME
    )
    logger.info("加载已有本地向量库")
    return vector_db
