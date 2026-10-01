# 后端：FastAPI + LangGraph Agent + 混合检索（CPU）。
# 镜像里只有依赖、代码和配置；embedding 模型、知识库（data/chunks）、向量库（chroma_db）运行时从宿主机只读挂载，
# 见 docker-compose.yml。容器内禁止联网下载模型（HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE，代码里也是 local_files_only）。
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# 国内网络可在 .env 里设 PIP_INDEX_URL（如清华镜像）；TORCH_INDEX_URL 指向 PyTorch 的 CPU 版源：
# PyPI 上的 Linux 版 torch 会带上 CUDA 依赖，镜像要大好几 GB，这里没有 GPU 用不上
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --index-url "$TORCH_INDEX_URL" "$(grep -E '^torch==' requirements.txt)" \
 && pip install --index-url "$PIP_INDEX_URL" -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY scripts ./scripts
COPY docker/api-entrypoint.sh /usr/local/bin/api-entrypoint.sh
RUN chmod +x /usr/local/bin/api-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["api-entrypoint.sh"]
CMD ["uvicorn", "api:app", "--app-dir", "src/s5_app", "--host", "0.0.0.0", "--port", "8000"]
