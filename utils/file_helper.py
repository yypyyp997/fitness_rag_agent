import os
import json
from utils import get_logger

logger = get_logger(__name__)

def ensure_dir(path: str):
    """如果文件夹不存在，自动创建文件夹"""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"创建目录: {path}")

def save_json(file_path: str, data):
    """保存json文件"""
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON已保存至: {file_path}")

def load_json(file_path: str):
    """读取json文件"""
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        raise FileNotFoundError(f"{file_path} 不存在")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
