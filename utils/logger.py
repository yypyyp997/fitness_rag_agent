import logging
import os
from config import settings

def get_logger(name: str, log_file: str = "run.log"):
    """
    全局日志生成函数
    :param name: 调用日志的模块名，例：__name__
    :param log_file: 日志文件名，默认写入 logs/run.log；传入如 "api.log" 可写到独立文件
    :return: logger对象
    """
    # 自动创建日志文件夹
    os.makedirs(settings.LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 日志格式：时间 - 日志级别 - 模块名 - 信息
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # 文件输出，日志写入logs目录
    file_handler = logging.FileHandler(
        os.path.join(settings.LOG_DIR, log_file),
        encoding="utf-8"
    )
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger
