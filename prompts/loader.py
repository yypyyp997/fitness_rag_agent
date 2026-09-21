from pathlib import Path

import yaml

from utils import get_logger

logger = get_logger(__name__)

# 路径以本文件位置锚定，从任何目录运行项目都能找到
PROMPTS_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROMPTS_DIR.parent / "config" / "prompt_config.yaml"

_config_cache = None

def _get_config() -> dict:
    """读取并缓存 prompt_config.yaml，整个项目只读一次"""
    global _config_cache
    if _config_cache is None:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"提示词配置文件不存在: {CONFIG_PATH}")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f) or {}
        logger.info(f"提示词配置加载完成: {CONFIG_PATH.name}")
    return _config_cache

def get_prompt(name: str) -> str:
    """
    按配置读取提示词全文
    :param name: prompt_config.yaml 中 prompt_files 下的键名
                 （rag_prompt / agent_system / tool_desc）
    说明：提示词 md 文件里以 # 开头的行视为注释，加载时自动剔除
    """
    config = _get_config()
    file_name = config.get("prompt_files", {}).get(name)
    if not file_name:
        raise KeyError(f"prompt_config.yaml 的 prompt_files 里没有配置: {name}")

    prompt_path = PROMPTS_DIR / file_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"提示词文件不存在: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 剔除 # 开头的注释行，正文直接作为提示词
    content = "".join(line for line in lines if not line.lstrip().startswith("#"))
    logger.info(f"加载提示词: {name} <- {prompt_path.name} ({len(content)}字)")
    return content.strip()

def get_generation_params() -> dict:
    """读取大模型生成参数（temperature 等），调参直接改 prompt_config.yaml"""
    return _get_config().get("generation", {}) or {}
