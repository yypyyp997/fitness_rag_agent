"""评估脚本：对编排系统跑评估集，输出三项指标 + JSON报告。

用法（项目根目录下）：
    python evaluation/eval_rag.py

评估维度：
- 路由准确率：实际路由 == expect_route（检验分流是否正确）
- 工具符合率：expect_tools 全部出现在实际 tools_used 中即算通过
  （包含关系而非精确相等：模型自主决定调用次数，比如计划题可能多检索一次）
- 关键词命中率：expect_keywords 在回答中的命中比例
  （direct题的核心检验：回答是否真的基于知识库内容生成）

输出：
- 控制台：逐题结果 + 汇总指标
- evaluation/results/eval_时间戳.json：完整报告（含每题回答原文，便于复盘）
- 退出码：全部通过为0，否则1（方便以后接CI）

设计说明：沿用 fitness_rag_basic 消融实验的"关键词命中"评估思路，
扩展到编排层（路由/工具是Agent项目特有的评估维度）。
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 支持 python evaluation\eval_rag.py 直接运行：把项目根加入模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台中文防乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "test_questions.json"
RESULTS_DIR = EVAL_DIR / "results"

from config import settings  # 路径处理完成后才能导入项目模块

def load_cases() -> list:
    """加载评估集"""
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        return json.load(f)

def check_case(case: dict, result: dict) -> dict:
    """单题检查：路由 / 工具 / 关键词"""
    answer = result.get("answer", "")
    route_ok = result.get("route") == case.get("expect_route")
    tools_ok = all(t in result.get("tools_used", []) for t in case.get("expect_tools", []))
    keywords = {kw: (kw in answer) for kw in case.get("expect_keywords", [])}
    keywords_ok = all(keywords.values()) if keywords else True
    return {
        "route_ok": route_ok,
        "tools_ok": tools_ok,
        "keywords": keywords,
        "keywords_ok": keywords_ok,
        "passed": route_ok and tools_ok and keywords_ok,
    }

def run_eval(cases: list) -> dict:
    """逐题执行编排入口并检查"""
    from agent import run_orchestrator  # 延迟导入：确保先完成向量库检查

    details = []
    total_start = time.perf_counter()
    for i, case in enumerate(cases, 1):
        start = time.perf_counter()
        result = run_orchestrator(case["question"])
        latency = round(time.perf_counter() - start, 2)
        check = check_case(case, result)
        detail = {
            **case,
            "actual_route": result.get("route"),
            "actual_tools": result.get("tools_used", []),
            "latency_sec": latency,
            "answer": result.get("answer", ""),
            **check,
        }
        details.append(detail)

        mark = "PASS" if detail["passed"] else "FAIL"
        hit = sum(detail["keywords"].values())
        total_kw = len(detail["keywords"])
        print(f"[{i}/{len(cases)}] {mark} 路由 {detail['expect_route']}->{detail['actual_route']} "
              f"工具 {detail['actual_tools']} 关键词 {hit}/{total_kw} 耗时{latency}s")
        print(f"      问题: {case['question']}")

    duration = round(time.perf_counter() - total_start, 1)
    total = len(details)
    all_kw = [k for d in details for k in d["keywords"].values()]
    summary = {
        "total": total,
        "pass_count": sum(d["passed"] for d in details),
        "route_accuracy": round(sum(d["route_ok"] for d in details) / total, 3),
        "tool_accuracy": round(sum(d["tools_ok"] for d in details) / total, 3),
        "keyword_hit_rate": round(sum(all_kw) / len(all_kw), 3) if all_kw else 1.0,
        "avg_latency_sec": round(sum(d["latency_sec"] for d in details) / total, 2),
        "duration_sec": duration,
    }
    return {"summary": summary, "cases": details}

def main():
    # 向量库前置检查（评估依赖检索）
    if not (os.path.exists(settings.CHROMA_PERSIST_DIR) and os.listdir(settings.CHROMA_PERSIST_DIR)):
        print("向量库不存在，请先运行 python main.py 完成构建")
        sys.exit(1)

    cases = load_cases()
    print(f"评估开始：共{len(cases)}题，模型 {settings.LLM_MODEL_NAME}\n")
    report = run_eval(cases)

    s = report["summary"]
    print("\n========== 评估汇总 ==========")
    print(f"通过: {s['pass_count']}/{s['total']}")
    print(f"路由准确率: {s['route_accuracy']:.1%}")
    print(f"工具符合率: {s['tool_accuracy']:.1%}")
    print(f"关键词命中率: {s['keyword_hit_rate']:.1%}")
    print(f"平均耗时: {s['avg_latency_sec']}s/题，总耗时 {s['duration_sec']}s")

    # 完整报告落盘（含每题回答原文）
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report["meta"] = {
        "timestamp": datetime.now().isoformat(),
        "model": settings.LLM_MODEL_NAME,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out_path}")

    sys.exit(0 if s["pass_count"] == s["total"] else 1)

if __name__ == "__main__":
    main()
