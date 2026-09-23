# # # import os
# # # from rag import prepare_chunks, build_vector_store, get_vector_store, get_retriever
# # # from config import settings
# # #
# # # # 1. 向量库已存在则直接加载（避免重复向量化）；否则先取chunk再构建
# # # if os.path.exists(settings.CHROMA_PERSIST_DIR) and os.listdir(settings.CHROMA_PERSIST_DIR):
# # #     print("检测到已有向量库，跳过构建")
# # #     db = get_vector_store()
# # # else:
# # #     # 获取chunk（自动走processed缓存）
# # #     chunks = prepare_chunks()
# # #     # 构建向量库并持久化（需要重建时手动删除 chroma_db 目录）
# # #     db = build_vector_store(chunks)
# # #
# # # # 2. 获取检索器
# # # retriever = get_retriever()
# # #
# # # # 3. 测试召回
# # # res = retriever.invoke("深蹲的注意事项")
# # # print(res)
# # from prompts import get_prompt, get_generation_params
# # from model.factory import get_llm
# # print(get_prompt('rag_prompt')[:50])
# # print(get_generation_params())
# # llm = get_llm()   # 构造成功即可，不用调用
# # print('LLM OK')
# from tools import get_tools, calculator
# print([t.name for t in get_tools()])          # ['rag_search', 'calculator', 'search_web']
# print(calculator.invoke({'expression': '(80 * 3) / 2'}))   # 计算结果: 120.0
import os
from agent import run_orchestrator
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
   健身知识问答 Agent（提交点6：命令行验证版）
--------------------------------------------------
   可问：知识库问题 / 训练计划 / 数学计算
   输入 exit / quit / 退出 可结束对话
==================================================
"""

def main():
    _ensure_vector_store()

    # 延迟导入：等向量库就绪后再初始化Agent（rag_search依赖检索器）
    from agent import run_agent

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

        print("\nAgent思考中...\n")
        try:
            answer = run_agent(question)
        except Exception as e:
            print(f"出错了：{e}\n（详细日志见 logs/run.log）")
            continue
        print(f"\n{answer}")
    r = run_orchestrator('帮我精确计算 (80*3)/2');
    print('路由:', r['route'], '| 工具:', r['tools_used']);
    print(r['answer'])
    r = run_orchestrator('深蹲的注意事项有哪些？');
    print('路由:', r['route'], '| 工具:', r['tools_used']);
    print(r['answer'][:200])
if __name__ == "__main__":
    main()
