# ─────────────────────────────────────────────────────
# BYOB-RAG – baked ALL-IN image (models included)
# multi-arch for linux/amd64 and linux/arm64
# ─────────────────────────────────────────────────────
ARG PY_VER=3.12
FROM python:${PY_VER}-slim

# native libs for faiss / llama-cpp
RUN apt-get update && apt-get install -y \
        build-essential cmake git curl \
        libopenblas-dev libomp-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    LLAMA_MODEL_PATH=/app/models/llama/Meta-Llama-3-8B-Instruct-Q8_0.gguf \
    BGE_EMBED_DIR=/app/embeddings/bge-large-en-v1.5 \
    MSMARCO_MODEL_DIR=/app/reranker/ms-marco-MiniLM-L-12-v2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy EVERYTHING (code + models + embeddings + reranker)
COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "byob_final.py"]
