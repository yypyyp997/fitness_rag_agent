# ===== 基础镜像：与本地开发一致的 Python 3.11 =====
FROM python:3.11-slim

WORKDIR /app

# ===== 先只复制依赖清单再安装：利用层缓存 =====
# 代码改动时不必重装依赖，构建只在依赖变化时才走这层
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ===== 复制项目代码（.dockerignore 在这里生效）=====
COPY . .

# ===== 声明端口（文档性质，真正的映射靠 run -p）=====
EXPOSE 8000

# ===== 启动命令：与 api/main.py 文档中的启动方式一致 =====
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]