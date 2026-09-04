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

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy llama-server and required shared libraries
COPY --from=llama-builder \
    /build/llama.cpp/build/bin/llama-server \
    /usr/local/bin/llama-server

# Install Python dependencies, excluding PyTorch first
COPY requirements.txt /app/requirements.txt

RUN grep -v -E '^(torch|torchvision)([<>=!~]|$)' \
        /app/requirements.txt \
        > /app/requirements-docker.txt

RUN python -m pip install \
    --no-cache-dir \
    -r /app/requirements-docker.txt

# CPU-only PyTorch for the CPU DigitalOcean server
RUN python -m pip install \
    --no-cache-dir \
    torch==2.11.0 \
    torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY . /app

RUN chmod +x /app/start.sh

EXPOSE 8080

CMD ["/app/start.sh"]
