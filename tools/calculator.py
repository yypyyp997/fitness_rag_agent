import math

from langchain_core.tools import tool

# 白名单命名空间：只暴露数学能力，内建函数全部置空，避免eval执行任意代码
# 说明：学习项目用白名单eval足够；生产环境建议换成numexpr或AST解析
_SAFE_NAMESPACE = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "int": int, "float": float,
    "pi": math.pi, "e": math.e,
    "sqrt": math.sqrt, "pow": math.pow, "exp": math.exp, "log": math.log,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "ceil": math.ceil, "floor": math.floor, "factorial": math.factorial,
}

@tool
def calculator(expression: str) -> str:
    """精确计算数学表达式。输入纯数学表达式，支持 + - * / ** % () 以及 sqrt/pow/log 等数学函数。涉及重量、组数、热量等数字计算时必须使用本工具，禁止心算。"""
    try:
        result = eval(expression, {"__builtins__": {}}, _SAFE_NAMESPACE)
        return f"计算结果: {result}"
    except Exception as e:
        # 报错信息会回给大模型，引导它下一次传纯数学表达式
        return f"计算失败: {str(e)}。请输入纯数学表达式，例如 (80 * 3) / 2"
