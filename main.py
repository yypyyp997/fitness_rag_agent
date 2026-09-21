# import os
# from rag import prepare_chunks, build_vector_store, get_vector_store, get_retriever
# from config import settings
#
# # 1. 向量库已存在则直接加载（避免重复向量化）；否则先取chunk再构建
# if os.path.exists(settings.CHROMA_PERSIST_DIR) and os.listdir(settings.CHROMA_PERSIST_DIR):
#     print("检测到已有向量库，跳过构建")
#     db = get_vector_store()
# else:
#     # 获取chunk（自动走processed缓存）
#     chunks = prepare_chunks()
#     # 构建向量库并持久化（需要重建时手动删除 chroma_db 目录）
#     db = build_vector_store(chunks)
#
# # 2. 获取检索器
# retriever = get_retriever()
#
# # 3. 测试召回
# res = retriever.invoke("深蹲的注意事项")
# print(res)
from prompts import get_prompt, get_generation_params
from model.factory import get_llm
print(get_prompt('rag_prompt')[:50])
print(get_generation_params())
llm = get_llm()   # 构造成功即可，不用调用
print('LLM OK')
