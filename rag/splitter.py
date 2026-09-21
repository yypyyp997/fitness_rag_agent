from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings
from utils import get_logger

logger = get_logger(__name__)


def split_documents(raw_docs):
    """
    递归字符文本分割，参数从settings读取
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len
    )
    split_docs = splitter.split_documents(raw_docs)
    logger.info(f"文档切分完成，chunk总数: {len(split_docs)}")
    return split_docs
