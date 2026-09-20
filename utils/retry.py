import time
from functools import wraps
from utils import get_logger

logger = get_logger(__name__)

def retry_decorator(max_retry: int = 3, sleep_sec: int = 2):
    """
    函数异常重试装饰器，调用大模型网络报错自动重试
    :param max_retry: 最大重试次数
    :param sleep_sec: 每次重试间隔秒数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_retry:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    logger.warning(f"执行失败，第{attempt}/{max_retry}次重试，错误：{str(e)}")
                    if attempt >= max_retry:
                        logger.error("达到最大重试次数，任务失败")
                        raise e
                    time.sleep(sleep_sec)
            return None
        return wrapper
    return decorator
