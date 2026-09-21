# utils/retry.py
import time
from functools import wraps
from utils.logger import get_logger

logger = get_logger(__name__)

def retry_decorator(max_retry: int = 3, sleep_sec: int = 1):
    """
    接口调用重试装饰器：调用大模型等网络接口报错时自动重试
    :param max_retry: 最大尝试次数
    :param sleep_sec: 每次重试间隔秒数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = max(max_retry, 1)
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"调用失败，第{attempt}/{attempts}次，错误：{str(e)}")
                    if attempt >= attempts:
                        logger.error("达到最大重试次数，任务失败")
                        raise  # 保留原始异常信息，方便定位问题
                    time.sleep(sleep_sec)
        return wrapper
    return decorator
