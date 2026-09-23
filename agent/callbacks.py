"""token用量回调：与中间件分工——中间件管工具和提示词，回调管模型token统计"""
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from utils import get_logger

logger = get_logger(__name__)

def _extract_token_usage(response) -> str:
    """从LLM返回结果里尽力提取token用量（不同模型字段位置不同）"""
    try:
        # ChatTongyi：用量一般在 llm_output 的 token_usage
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage")
        if usage:
            return f"输入{usage.get('input_tokens', '?')}tok / 输出{usage.get('output_tokens', '?')}tok"
        # 兜底：从generation_info里翻
        for generation_list in getattr(response, "generations", []):
            for generation in generation_list:
                info = getattr(generation, "generation_info", None) or {}
                usage = info.get("token_usage") or info.get("usage")
                if usage:
                    return f"输入{usage.get('input_tokens', '?')}tok / 输出{usage.get('output_tokens', '?')}tok"
    except Exception:
        pass
    return "用量未知"

class TokenUsageHandler(BaseCallbackHandler):
    """模型token用量统计（写入run.log，提交点9评估的成本数据来源）"""

    def on_llm_end(self, response, **kwargs: Any) -> None:
        logger.info(f"<—— 模型输出完成（{_extract_token_usage(response)}）")
