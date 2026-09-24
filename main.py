import os

from config import settings

def _ensure_vector_store():
    """向量库不存在时自动构建（首次运行/删除chroma_db后需要）"""
    if os.path.exists(settings.CHROMA_PERSIST_DIR) and os.listdir(settings.CHROMA_PERSIST_DIR):
        return
    print("首次运行：正在构建向量库（约1-2分钟）...")
    from rag import prepare_chunks, build_vector_store
    build_vector_store(prepare_chunks())
    print("向量库构建完成")

WELCOME = """
==================================================
   健身知识问答 Agent（提交点8：编排入口版）
--------------------------------------------------
   简单知识问答走RAG直答，复杂任务走完整Agent循环
   可问：知识库问题 / 训练计划 / 数学计算 / 联网资讯
   输入 exit / quit / 退出 可结束对话
==================================================
"""

def main():
    _ensure_vector_store()

    # 延迟导入：等向量库就绪后再加载编排工作流（rag_search依赖检索器）
    from agent import run_orchestrator

    print(WELCOME)
    while True:
        try:
            question = input("\n你的问题 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q", "退出"):
            print("再见！")
            break

        print("\n处理中...\n")
        try:
            result = run_orchestrator(question)
        except Exception as e:
            print(f"出错了：{e}\n（详细日志见 logs/run.log）")
            continue

        # 回答前先展示路由元信息，直观呈现编排层的分流决策
        tools = "、".join(result["tools_used"]) if result["tools_used"] else "无"
        print(f"【路由: {result['route']} | 工具: {tools}】\n")
        print(result["answer"])

if __name__ == "__main__":
    main()
