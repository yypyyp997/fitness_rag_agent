from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)
from utils import get_logger

logger = get_logger(__name__)

def load_documents(data_dir: Path):
    """
    加载raw目录下多格式文档：md/txt/pdf/docx
    md直接用TextLoader读取纯文本，轻量无额外依赖
    """
    docs = []
    if not data_dir.exists():
        logger.warning(f"文档目录不存在: {data_dir}")
        return docs

    file_map = {
        "*.md": TextLoader,
        "*.txt": TextLoader,
        "*.pdf": PyPDFLoader,
        "*.docx": Docx2txtLoader
    }

    for glob_pattern, loader_cls in file_map.items():
        for file_path in data_dir.glob(glob_pattern):
            try:
                if loader_cls is TextLoader:
                    # 只有TextLoader支持encoding参数，指定utf-8防止中文乱码
                    loader = loader_cls(str(file_path), encoding="utf-8")
                else:
                    # PyPDFLoader / Docx2txtLoader 不接受encoding参数，直接传路径
                    loader = loader_cls(str(file_path))
                page_docs = loader.load()
                docs.extend(page_docs)
                logger.info(f"成功加载文档: {file_path.name}, 页数/片段数:{len(page_docs)}")
            except Exception as e:
                logger.error(f"加载文件失败 {file_path.name}, error: {str(e)}")
    logger.info(f"文档加载完成，总原始文档数量: {len(docs)}")
    return docs
