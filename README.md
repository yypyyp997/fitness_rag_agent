# Fitness RAG Agent

基于 **LangChain + LangGraph + 通义千问（DashScope）+ Chroma + FastAPI + Streamlit** 构建的健身领域 RAG 智能体。

一句话说清它：**简单知识问答走 RAG 直答支路（1 次检索 + 1 次生成，快且省 token），复杂任务走完整 ReAct Agent 循环（工具自主调度），LangGraph 编排层统一分流；对外提供 Web 界面与标准 API 双入口，支持多轮对话记忆与会话管理**。

> 基线项目：fitness_rag_basic（纯 RAG 问答链 + 检索参数消融实验）。本项目在其基础上完成 Agent 化与服务化改造：路由编排、工具调用、流式输出、FastAPI 接口层、SQLite 多轮记忆、多会话管理、评估体系、单元测试。每个提交点都有真实 API 实测验证，踩坑记录见第 9 节。

## 1. 功能总览

| 能力 | 说明 |
|---|---|
| 健身知识问答 | 本地向量库检索增强作答，回答附来源片段标注，覆盖训练/饮食/恢复/伤病四大域 |
| 智能路由分流 | LLM 单词分类（direct/agent），判错也安全（agent 是通用兜底路径） |
| 训练计划制定 | 关键词触发的"计划模式"动态系统提示词，多轮检索 + 综合输出 |
| 数学计算 | 白名单安全计算器，`__import__`/`open` 等危险调用一律拦截 |
| 联网搜索 | Tavily 工具（未配 Key 时优雅降级为文字提示，不抛异常） |
| **FastAPI 服务层** | 四个 HTTP 接口：健康检查 / 非流式问答 / SSE 流式问答 / 会话删除，/docs 自动交互文档，lifespan 启动预热 |
| **多轮对话记忆** | SQLite 全量存储 + 滑动窗口（最近 6 条）注入到路由、直答、Agent 三处 |
| **会话管理** | 新建 / 切换 / 删除 / 清空，首问自动命名，历史会话可回放 |
| 双入口 | Streamlit 网页（流式打字机 + 会话列表）+ API 服务（任何 HTTP 客户端可调） |
| 评估体系 | 路由/工具/关键词三指标 + JSON 报告落盘 + CI 友好退出码 |
| 单元测试 | 49 例纯逻辑单测，0.02 秒跑完，零 API 依赖 |

## 2. 系统架构：双入口，共享业务核心

```
   入口A：Streamlit 网页（:8501）        入口B：HTTP 客户端（:8000）
   app.py 进程内函数直连                 api/main.py 协议转换（HTTP/SSE）
   流式打字机 + 会话列表 UI               /api/chat · /api/chat/stream ...
          │                                    │
          └────────────────┬───────────────────┘
                           ▼
            agent/agent_orchestrator.py（LangGraph 编排核心）
                           │
              ┌────────────▼─────────────┐
              │   router 路由节点          │  LLM 单词分类（qwen-turbo）
              │   失败兜底走 agent         │  只输出 direct / agent 一个词
              └────────────┬─────────────┘
            ┌──────────────┴──────────────┐
    direct（知识问答）                agent（复杂任务）
  ┌───────────▼───────────┐   ┌───────────▼────────────────┐
  │ rag_search 向量检索top3  │   │ preprocess 输入清洗/截断     │
  │ 历史块拼在检索模板前      │   │ ReAct 循环（create_agent）    │
  │ rag_prompt 单次生成      │   │ ├─ rag_search（多轮可调用）  │
  │ dashscope原生SDK 真流式  │   │ ├─ calculator（白名单安全）  │
  └───────────┬───────────┘   │ ├─ search_web（Tavily降级）  │
              │               │ └─ 计划模式动态提示词          │
              │               │ postprocess 倒序取最终回答     │
              │               │ 完整循环后按块渐进输出          │
              │               └───────────┬────────────────┘
              └──────────────┬────────────┘
                    ┌────────▼─────────┐
                    │  回答 + 元信息     │   route / tools_used / latency
                    │  有session则落库   │   SQLite: data/memory.sqlite3
                    └──────────────────┘
```

### 2.1 为什么要编排层

简单知识问答走完整 Agent 循环要 2~3 次 LLM 调用 + 工具往返（约 5~8 秒）；直答支路 1 次检索 + 1 次生成（约 2~4 秒），token 消耗约为前者的三分之一。路由器只需"大概对"——即使判错，agent 通用路径也能正确回答，代价只是多花 token，不会答不了。这个"兜底哲学"让路由提示词可以保持极简（单词输出、零 few-shot）。

### 2.2 FastAPI 起什么作用（与 Streamlit 的关系）

**两个入口是并列关系，不是依赖关系**：Streamlit 进程内直接 import 调用编排函数（不过网络）；FastAPI 是独立进程，把同样的能力暴露成 HTTP 接口。两者互不知道对方存在——判据很硬：agent 包里没有一个 `import fastapi`，api 包里没有一个 `import streamlit`。这就是"业务核心与入口解耦"。

FastAPI 在项目里承担五件事：

| 职责 | 落点 |
|---|---|
| 协议转换（核心） | 把 `run_orchestrator()` / `stream_orchestrator()` 变成 HTTP 可调接口——能力从此不锁在 Python 进程里 |
| Web 路由 | URL → 处理函数映射（`@app.post("/api/chat")`） |
| 请求校验 | pydantic 自动校验（question 1~500 字、session_id 正则），不合法直接 422 |
| 生命周期 | lifespan 启动时预热向量库和编排图，首请求不背冷启动 |
| 文档与错误码 | /docs 自动交互文档；上游异常 502、校验失败 422、成功 200 |

**没有前端也值得做 FastAPI**：服务化的价值不取决于今天有没有界面。它让能力可以被任何系统调用（curl、评估脚本、CI、未来的网页前端），也是"前后端分离"架构的接缝证据——写一个消费 SSE 的 HTML 聊天页只要十几行 JS。

> 注意区分两个"路由"（面试高频坑）：**Web 路由**是 URL 到处理函数的映射（FastAPI，api 层）；**业务路由**是 LLM 判断问题走 direct 还是 agent 支路（LangGraph 条件边，agent 层）。同词不同义，层也不同。

## 3. 目录结构

```
fitness_rag_agent/
├── main.py                   # 命令行入口：首次运行自动建向量库
├── app.py                    # Streamlit 界面：流式输出 + 多会话管理
├── api/                      # ===== FastAPI 服务层 =====
│   ├── main.py               #   四接口 + lifespan 预热 + SSE 事件流
│   └── schemas.py            #   pydantic 请求/响应模型（自动校验）
├── agent/                    # ===== Agent 层 =====
│   ├── react_agent.py        #   create_agent 构建 ReAct 循环 + AgentContext
│   ├── middleware.py         #   官方中间件：工具监控/计划模式/输入输出处理
│   ├── callbacks.py          #   TokenUsageHandler 用量统计（非流式路径）
│   ├── memory.py             #   多轮记忆：SQLite 存储 + 滑动窗口 + 会话生命周期
│   └── agent_orchestrator.py #   LangGraph 编排：路由分流 + 双支路 + 双流式
├── rag/                      # ===== RAG 层 =====
│   ├── loader.py             #   多格式加载：md/txt/pdf/docx
│   ├── splitter.py           #   递归字符切分（300/30）
│   ├── vector_store.py       #   Chroma 持久化 + MD5 切分缓存（两层缓存）
│   ├── retrieval.py          #   检索器封装（top_k=3）
│   └── rerank.py             #   重排序（预留位，见踩坑 9.10 的启用条件）
├── tools/                    # ===== 工具层 =====
│   ├── rag_search.py         #   向量检索（缓存检索器 + 来源标注）
│   ├── calculator.py         #   白名单 eval 安全计算器
│   └── search_web.py         #   Tavily 搜索（无 Key 优雅降级）
├── prompts/                  # ===== 提示词模块 =====
│   ├── loader.py             #   yaml 配置 + .md 热加载（#注释行自动剔除）
│   ├── rag_prompt.md         #   RAG 直答模板（{context} {question}）
│   ├── agent_system.md       #   Agent 常规模式系统提示词
│   └── tool_desc.md          #   工具描述（计划模式增强素材）
├── model/factory.py          # 模型工厂：ChatTongyi / DashScopeEmbeddings（带重试）
├── config/
│   ├── settings.py           # 全局配置单例
│   └── prompt_config.yaml    # 生成参数（temperature 等，改参数不动代码）
├── evaluation/               # ===== 评估模块 =====
│   ├── eval_rag.py           #   三指标评估 + JSON 报告 + 退出码
│   ├── test_questions.json   #   评估集：8 题覆盖双支路
│   └── results/              #   报告落盘目录（gitignore）
├── tests/                    # 49 例单元测试（unittest，零 API 依赖）
│   ├── test_agent_tool.py    #   中间件/编排结构/计算器安全/评估逻辑/重试
│   ├── test_rag_retrieval.py #   RAG 配置边界/切分/提示词加载
│   └── test_memory.py        #   记忆存取/窗口/会话隔离/路径隔离回归
└── data/
    ├── raw/                  # 知识库四件套（pdf/md/txt/docx）
    ├── processed/            # 切分缓存（gitignore）
    └── memory.sqlite3        # 对话记忆库（gitignore，运行时自动创建）
```

## 4. 技术栈

| 组件 | 选型 | 实测版本 |
|---|---|---|
| 语言 | Python | 3.11.9 |
| LLM | 通义千问 qwen-turbo（DashScope） | dashscope 1.27.6 |
| Embedding | DashScope text-embedding-v4 | — |
| Agent 框架 | LangChain（create_agent + 官方 middleware） | langchain 1.4.1 |
| 编排 | LangGraph StateGraph | langgraph 1.2.11 |
| 向量库 | Chroma（本地持久化） | chromadb 1.5.9 |
| API 服务 | FastAPI + uvicorn + pydantic v2 | fastapi 0.141.1 / uvicorn 0.53.0 |
| 界面 | Streamlit（st.empty 流式重绘 + 会话列表） | 1.64.0 |
| 记忆存储 | SQLite（标准库 sqlite3，WAL 模式） | — |
| 文档解析 | PyPDFLoader / Docx2txtLoader / TextLoader | — |

## 5. 快速开始

### 环境要求

- Python 3.10+（开发环境 3.11.9）
- DashScope API Key（[获取地址](https://dashscope.console.aliyun.com/)）

### 安装

```bash
cd fitness_rag_agent
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 配置 API Key

项目根目录创建 `.env`（已被 .gitignore 忽略）：

```
DASHSCOPE_API_KEY=***
```

也可直接配系统环境变量。可选：注册 [Tavily](https://tavily.com/) 配置 `TAVILY_API_KEY` 启用联网搜索。

### 运行（四种方式）

```bash
streamlit run app.py                             # 网页界面（:8501，流式+多会话）
uvicorn api.main:app --host 0.0.0.0 --port 8000  # API 服务（:8000，/docs 自动文档）
python evaluation\eval_rag.py                    # 评估（8 题，约 30 秒）
python -m unittest discover tests -v             # 单元测试（49 例，0.02 秒）
```

首次运行会自动构建向量库（约 1~2 分钟）。

## 6. 知识库说明

`data/raw/` 为四件套结构，共切分 **64 个 chunks**：

| 文件 | 格式 | 内容域 |
|---|---|---|
| 健身基础理论与核心原理 | pdf | 理论根基：减脂/增肌/塑形/饮食原理 |
| 力量训练 + 有氧训练 完整权威手册 | md | 动作模式、自由重量/器械体系、LISS/HIIT、伤病预防 |
| 增肌专项、减脂专项、睡眠恢复、伤病管理 | txt | 热量/蛋白质标准、平台期、睡眠激素、伤病区分 |
| 训练计划设计体系 + 周期化训练 + 健身误区 + 高频答疑 | docx | 计划设计、周期化、误区答疑 |

多格式混布是有意设计：验证 loader 对四种真实格式的兼容性（中文+空格文件名、PDF 文本层、docx 表格文本均实测通过）。

**改知识库的正确姿势**（两层缓存失效行为不同，最高频的坑）：

1. 直接增删改 `data/raw/` 下文件
2. 切分缓存（`data/processed/`）带 MD5 校验，**自动**失效重切
3. 向量库（`chroma_db/`）**不会**自动重建——手动删除该目录后运行即重建
4. 评估集关键词若引用旧库表述，同步更新 `evaluation/test_questions.json`（原因见踩坑 9.5）

## 7. 使用示例

### 网页模式（Streamlit）

侧边栏"会话历史"区：**＋新建会话**（立即入库登记）、点击列表项**切换**（回放最近 25 轮）、**🗑 删除**会话；会话标题取首条提问的前 20 个字符，一眼可辨。对话区流式输出，回答下方展示 `路由 / 工具 / 耗时` 元信息。

多轮追问实测（记忆生效的铁证）：

```
你：我身高175体重80，我的bmi是多少？
AI：你的BMI（身体质量指数）为26.12，属于超重范围……（路由 agent · 工具 calculator）
你：那我这样的身高体重应该怎么增肌减脂？
AI：根据你的身高175cm、体重80kg的体型，结合增肌减脂的目标……
    （第二问没带任何数字，回答精准引用了第一轮的数据——多轮记忆生效）
```

### API 模式（FastAPI）

```bash
# 健康检查
curl http://localhost:8000/api/health

# 非流式问答（同一 session_id 两连问 = 多轮）
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
     -d '{"question": "我身高175体重80，我的bmi是多少？", "session_id": "demo"}'

# 流式问答（SSE：delta 事件逐块推送，meta 收尾，done 结束）
curl -N -X POST http://localhost:8000/api/chat/stream -H "Content-Type: application/json" \
     -d '{"question": "深蹲的注意事项", "session_id": "demo"}'

# 删除会话
curl -X DELETE http://localhost:8000/api/sessions/demo
```

SSE 事件协议（对齐 OpenAI/Anthropic 流式风格）：

```
event: delta   data: {"text": "增量文本"}                                  （多次）
event: meta    data: {"route": "...", "tools_used": [...], "session_id": "demo",
                      "latency_sec": 3.2}                                  （1次）
event: done    data: [DONE]
```

交互式文档：浏览器打开 `http://localhost:8000/docs`，所有接口可直接填参执行。

## 8. 多轮记忆与会话管理

**存储与注入解耦**（核心设计）：

- **存储层**（agent/memory.py）：SQLite 按 session 全量落库，可审计可恢复。每次操作独立连接 + `check_same_thread=False`（适配 uvicorn 线程池）+ WAL 模式（读写并发友好）+ session_id 索引
- **注入层**：调用编排层时按需取最近 6 条（约 3 轮，单条再截 200 字），注入到**三处**——路由提示词前（让"那硬拉呢"这类追问能被正确分类）、direct 支路检索模板前（回答衔接上文）、Agent 支路 messages 列表（Human/AIMessage 交替）
- **落库时机**：回答成功生成后才写 user + assistant 两条，失败不污染历史

**会话生命周期**（ChatGPT 式体验）：

| 动作 | 行为 |
|---|---|
| 点击"＋新建会话" | `create_session` **立即登记入库**（空会话也出现在列表），标题默认"新会话" |
| 首条提问 | 自动命名：提问前 20 字作为标题（如"我身高175体重80，我的bmi是多少"） |
| 点击列表项切换 | `get_recent_messages(limit=50)` 回放最近 25 轮到界面 |
| 点 🗑 删除 | `delete_session` 消息 + 元信息一起删，列表不再出现 |
| "清空本会话记忆" | `clear_session` 只清消息保留会话壳，允许重新自动命名 |

会话列表查询用 UNION 同时覆盖两类来源：已登记但还没消息的新会话、只有消息没有元信息行的旧格式会话（向前兼容）。

**为什么不用 LangChain 自带 Memory 类**：它把"存储"与"注入"耦在一起不好控制；解耦后换存储后端（SQLite → Postgres）只改 memory.py 一个文件。窗口取 6 条是质量/成本平衡——更长窗口提升有限但 token 线性上涨；摘要压缩、语义检索历史是后续迭代方向。

## 9. 踩坑记录（真实排障过程）

### 9.1 ChatTongyi.stream() 假流式

**现象**：界面 250 字回答"一次性蹦出来"，没有打字机效果。
**诊断**：最小示例逐块计数——LangChain 封装层只返回 **1 个 chunk**（整流被聚合），dashscope 原生 SDK 同请求返回 **52 个增量块**。
**解决**：direct 支路直连 `dashscope.Generation.call(stream=True, incremental_output=True)`，从 `chunk.output.choices[0].message.content` 取增量，实测 603 字回答 104 个增量块。
**教训**：框架封装层可能吞掉底层能力。流式不生效先用最小示例验证**底层 SDK**，再定责。

### 9.2 temperature 传参不生效

`ChatTongyi(temperature=0.7)` 直接传参不生效，必须走 `model_kwargs={"temperature": 0.7}` 才透传给 DashScope。已固化在 model/factory.py，参数从 yaml 热加载。

### 9.3 两层缓存失效不同步（最高频的坑）

切分缓存（MD5 校验）改库自动失效；向量库永久持久化**不会**自动重建——改了知识库后切分缓存重建了、检索结果却纹丝不动。解法：改库后手动删 `chroma_db/` 再运行。

### 9.4 路由边界误判（评估驱动修复）

初版路由提示词写"简单/复杂"这种不可判定描述，评估 **6/8**："热量缺口多少合适"、"腰刺痛怎么办"被误判到 agent。修复：写出可判定边界——"含具体数字（摄入量/热量缺口/休息时长）归 direct，数学计算（**含算式**）才归 agent"。复测 **8/8**。教训：LLM 分类器的边界必须可判定，且要靠评估集守护。

### 9.5 评估集与知识库强耦合（换库即假阴性）

知识库从单文件 md 换成四件套后，评估掉到 **5/8**。路由 100%、工具 100%，挂的全是关键词——评估集还写着旧库表述（旧库"膝盖、呼吸"，新库写"膝盖内扣"没提呼吸）。逐题核实三轮真实回答原文后改关键词，复测 8/8。**教训：换知识库 = 换了被测系统的一部分；关键词必须从当前库的真实回答里选。**

### 9.6 Windows 控制台中文乱码

控制台默认 GBK。脚本入口 `sys.stdout.reconfigure(encoding="utf-8")` 或设 `PYTHONIOENCODING=utf-8`；TextLoader 显式 `encoding="utf-8"`。

### 9.7 工具异常要优雅降级，不要抛给循环

工具 traceback 进入上下文，模型会把报错当资料继续推理。`search_web` 无 Key 时返回自然语言提示让模型自己解释——**工具的"失败"也是有效输出**。

### 9.8 多格式知识库的格式细节

扫描版 PDF 是哑弹（PyPDFLoader 只抽文本层，空内容"建库成功"但检索不到——建库后必须抽查 chunk）；导出工具的 md 常带 `\+` 转义符轻微污染检索文本；docx 表格结构会丢失。

### 9.9 工具符合率用包含关系，不用精确相等

模型自主决定调用次数（计划题可能检索两次），精确相等产生大量假阴性。判定规则：**期望工具 ⊆ 实际工具**，有对应单元测试。

### 9.10 rerank 为什么没启用（负决策也是决策）

知识库仅 64 chunks，向量检索直接取 top3 精度足够；rerank 解决的是大候选池（粗召回 top20~50）的精排问题，当前规模上属于过度设计。`rag/rerank.py` 预留位保留，等库扩容后用评估集做有无对比再启用——**知道什么时候不做，和知道怎么做同样重要**。

### 9.11 模块级常量锁定路径 → 测试隔离失效（两例失败 + 真实库污染）

**现象**：跑测试 2 例失败，且测试会话数据写进了真实的 `data/memory.sqlite3`。
**根因**：memory 库路径原是模块级常量，**导入时**就锁定。unittest 按字母序先跑 test_agent_tool → 连带导入 agent 包 → memory 用默认路径完成初始化 → test_memory 后设的 `MEMORY_DB_PATH` 环境变量形同虚设。
**修复**：路径改 `_db_path()` 函数内惰性读取 + 新增回归测试硬断言"路径必须跟随环境变量"。
**教训**：模块级副作用（导入时读环境变量/锁路径/建连接）是测试隔离的天敌——配置要在**使用时**读取，不是**导入时**。`python -m unittest` 按字母序发现测试，导入顺序是不可控的。

### 9.12 Streamlit 全脚本重跑 → 定义位置就是运行时顺序

**现象**：点击"清空本会话记忆"按钮报 NameError，但 py_compile 编译通过。
**根因**：`_session_id()` 定义在侧边栏代码之后，而侧边栏按钮在函数定义前就调用它。Streamlit 每次交互**从头重跑整个脚本**——函数定义位置就是执行顺序，"定义晚于使用"从静态问题变成了运行时问题（语法检查查不出来）。
**修复**：定义提到所有使用点之前。
**教训**：Streamlit 的执行模型是"每次交互全量重跑"，不是"注册一次到处用"的事件模型。

### 9.13 非 ASCII 字符的补丁管线损坏（孤立代理对）

**现象**：通过脚本给 app.py 打含 emoji（🗑）的小补丁，两次都把文件写坏（emoji 被拆成 UTF-16 孤立代理对写入），编译错误信息混乱难定位。
**修复**：放弃补丁路线**整文件重写** + 字节级检查（扫描 0xD800~0xDFFF 代理对区段）作为交付断言。
**教训**：① 对含非 ASCII 字符的文件做小修改，优先整文件重写而不是文本补丁；② 交付管线要有字节级自动检查；③ emoji 是锦上添花，工程上能不用就不用。

### 9.14 上游 API 网络瞬断（超时重试的活案例）

E2E 验证中一次计算请求 120 秒超时（平时 1.4 秒），5 秒后重试立即成功——dashscope 偶发瞬断真实存在。**教训**：外部依赖必须有超时预算和重试策略（项目里 embedding 构建挂了重试装饰器，LLM 调用的超时中间件在后续计划里）。

## 10. 评估体系

```bash
python evaluation\eval_rag.py
```

**三指标设计**：

| 指标 | 判定方式 | 检验什么 |
|---|---|---|
| 路由准确率 | 实际路由 == 期望路由 | 分流是否正确 |
| 工具符合率 | 期望工具 ⊆ 实际调用工具 | Agent 是否按需调度工具 |
| 关键词命中率 | 期望关键词在回答中的命中比例 | 回答是否真基于知识库生成 |

**当前实测**（qwen-turbo，8 题，30.4 秒）：通过 **8/8**，路由/工具/关键词均 **100%**，平均 **3.80 s/题**。历史轨迹：初版 6/8（路由误判）→ 修提示词 → 8/8；换知识库 5/8（假阴性）→ 同步评估集 → 8/8。两次完整的"评估发现问题 → 定位 → 修复 → 复测"闭环。

报告落盘 `evaluation/results/eval_时间戳.json`（含每题回答原文），退出码全过 0 / 否则 1，可直接接 CI。

**评估集维护原则**：改路由提示词、换知识库、调检索参数之后必须重跑；关键词从当前知识库的真实回答里选，不凭记忆写。

## 11. 测试

```bash
python -m unittest discover tests -v
```

49 例全绿，0.02 秒完成。选 unittest 而非 pytest：标准库零依赖 CI 不装包；pytest 完全兼容 unittest.TestCase，将来切换零改动。分工哲学：**单测守代码结构（0.02 秒），评估守系统行为（30 秒）**。

| 测试文件 | 覆盖点 | 数量 |
|---|---|---|
| test_agent_tool.py | 输入预处理 / 输出提取 / 计划关键词 / 路由兜底 / 图结构 / 计算器安全 / 评估判定 / 重试装饰器 / JSON 工具 | 30 |
| test_rag_retrieval.py | RAG 配置边界 / 文档切分 / 提示词加载与注释剔除 | 13 |
| test_memory.py | 记忆存取 / 滑动窗口 / 会话隔离 / 清空 / 历史格式化 / **路径隔离回归** | 6 |

技巧两则：轻量替身（postprocess 只读 type/content 两属性就用假对象替代 AIMessage）；图结构测试（`get_graph().nodes` 验证三节点存在，不调 LLM）。

## 12. 设计决策与思考

**① 编排双支路，而不是全走 Agent**。直答支路砍掉循环换速度；路由判错安全（兜底），提示词才敢极简。

**② 兜底方向永远朝通用路径**。路由失败→agent、输出不认识→agent、输入为空→拦截。所有异常路径收敛到"一定能给出回答"的那条路。

**③ 工具统计走 middleware 注入**。`@wrap_tool_call` 写入 AgentContext，编排层按引用回收；不侵入 LangGraph 状态、不 hack 框架内部。

**④ 流式的诚实设计**。direct 真流式（原生增量）；agent 工具决策轮无正文可流式，完整循环后按块渐进。界面体验一致，实现诚实区分"能流"与"不能流"。

**⑤ API 薄壳原则**。api 层只做协议转换零业务逻辑（判据：agent 包不 import fastapi）；业务改动永远不碰 Web 层，Web 层改造永远不碰业务层。

**⑥ 存储与注入解耦（记忆）**。SQLite 全量存储可审计可恢复；注入按 6 条窗口保证上下文经济性。换存储后端只改一个文件。

**⑦ 会话生命周期 ChatGPT 式**。新建立即登记（空会话可见）→ 首问自动命名 → 历史可回放 → 删除彻底。状态管理清晰，无"幽灵会话"。

**⑧ 检索参数继承消融结论**。300/30/top3 来自基线项目实验，不重复调参；出处写在配置注释里。

**⑨ 配置与代码分离的度**。业务常调的（temperature、系统提示词）进配置热加载；结构稳定的（路由提示词）放代码常量。不是所有提示词都该配置化。

## 13. 与基线项目对比

| 能力 | fitness_rag_basic | 本项目 |
|---|---|---|
| 知识库问答 | 单条 RAG 链 | 直答支路（来源标注 + 真流式） |
| 多轮工具调度 | 无 | ReAct 循环 + 官方 middleware |
| 路由分流 | 所有问题一条链 | LangGraph 双支路 + 兜底 |
| 数学计算 / 联网搜索 | 无 | calculator / search_web |
| 计划模式 | 无 | 动态系统提示词切换 |
| **API 服务化** | 无 | FastAPI 四接口 + SSE + lifespan 预热 |
| **多轮对话记忆** | 无 | SQLite + 滑动窗口三处注入 |
| **会话管理** | 无 | 新建/切换/删除/自动命名/回放 |
| 交互界面 | 命令行 | 命令行 + Streamlit + API |
| 评估 | 检索参数消融 | 三指标 + JSON 报告 + CI 退出码 |
| 测试 | 无 | 49 例单元测试 |
| 知识库 | 单文件 md | 四件套多格式（pdf/md/txt/docx） |

## 14. 后续计划

- [ ] 评估升级：LLM-as-judge / RAGAS（faithfulness / relevancy / context precision）
- [ ] 记忆升级：历史摘要压缩、按语义检索相关历史片段
- [ ] API 加固：访问令牌鉴权、限流、超时与重试中间件
- [ ] 容器化部署上公网（Docker Compose：api + 界面，投递前可选）
- [ ] 前端聊天页：消费 SSE 接口的纯 HTML/JS 实现
- [ ] 知识库扩容后启用 rerank（评估驱动 AB 对比，预留位已就绪）
- [ ] 多用户支持：JWT 鉴权 + 记忆库迁 Postgres + 服务无状态化
- [ ] Tavily 联网搜索启用（代码就绪，配置 Key 即用）

## 15. 一点方法论

项目按"提交点"逐步长出来：每步只加一个能力，交付前真实 API 验证再提交，git 历史就是开发叙事。坚持了三件事：

1. **每个坑都写进 README**——踩坑不可耻，重复踩才可耻；排障故事比"我会用 LangChain"值钱得多。
2. **评估先于优化**——先有 6/8 的度量，才有路由修复和 8/8 的复测；没有度量就没有优化。
3. **测试与评估分工**——单测 0.02 秒守结构，评估 30 秒守行为，两条线互补不重复。
