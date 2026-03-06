# ============================================================
# Dockerfile — YOLOv5 云台追踪系统
# 基础镜像：Jetson 用 nvcr.io/nvidia/l4t-pytorch；
#           x86 开发机用 pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
# ============================================================
ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

LABEL maintainer="zan"
LABEL description="YOLOv5 gimbal tracking system"

# ── 代理设置（构建阶段走 7897 端口） ──────────────────────────
ARG HTTP_PROXY=http://host.docker.internal:7897
ARG HTTPS_PROXY=http://host.docker.internal:7897
ARG NO_PROXY=localhost,127.0.0.1
ENV http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY}

# ── 系统依赖 ─────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        # OpenCV 运行时
        libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
        # I2C 工具
        i2c-tools \
        # 音频（蜂鸣器脚本可能需要）
        alsa-utils \
        # 其他
        git wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python 依赖 ───────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .

# pip 同样走代理
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── 拷贝项目代码 ──────────────────────────────────────────────
COPY . .

# ── 清除构建阶段代理（运行时不强制走代理） ────────────────────
ENV http_proxy="" \
    https_proxy="" \
    no_proxy=""

# ── 运行时默认命令 ────────────────────────────────────────────
CMD ["python3", "main.py"]
