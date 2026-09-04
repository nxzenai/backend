# =========================================================
# Stage 1 - Build llama.cpp
# =========================================================
FROM debian:bookworm-slim AS llama-builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libcurl4-openssl-dev \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN git clone --depth 1 \
    https://github.com/ggml-org/llama.cpp.git

WORKDIR /build/llama.cpp

RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
    && cmake --build build \
        --config Release \
        -j2 \
        --target llama-server


# =========================================================
# Stage 2 - NxZenAI backend
# =========================================================
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Baked Llama model location
ENV GENAI_MODEL_DIR=/opt/models

# llama.cpp runtime libraries
ENV LD_LIBRARY_PATH=/usr/local/lib/llama

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*


# =========================================================
# Copy llama-server and required shared libraries
# =========================================================
COPY --from=llama-builder \
    /build/llama.cpp/build/bin/ \
    /usr/local/lib/llama/

RUN ln -s /usr/local/lib/llama/llama-server \
    /usr/local/bin/llama-server


# =========================================================
# Install Python dependencies
# =========================================================
COPY requirements.txt /app/requirements.txt

RUN grep -v -E '^(torch|torchvision)([<>=!~]|$)' \
    /app/requirements.txt \
    > /app/requirements-docker.txt

RUN python -m pip install \
    --no-cache-dir \
    -r /app/requirements-docker.txt


# =========================================================
# Install CPU-only PyTorch
# =========================================================
RUN python -m pip install \
    --no-cache-dir \
    torch==2.11.0 \
    torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu


# =========================================================
# Download Fast Llama model during Docker build
# =========================================================
RUN mkdir -p /opt/models \
    && python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF', filename='llama-3.2-1b-instruct-q4_k_m.gguf', local_dir='/opt/models')"


# =========================================================
# Copy NxZenAI application
# =========================================================
COPY . /app

RUN chmod +x /app/start.sh


# =========================================================
# Application port
# =========================================================
EXPOSE 8080


# =========================================================
# Start FastAPI + local llama-server
# =========================================================
CMD ["/app/start.sh"]
