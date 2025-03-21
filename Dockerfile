# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

# ARG PYTHON_VERSION=3.9.6
# FROM python:${PYTHON_VERSION}-slim as base

# BOOTSTRAP docker

FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    python3-opencv \
    libglib2.0-0 \
    nvidia-container-toolkit \
    python3 -m pip install --upgrade pip \
    python3 -m pip install -r requirements.txt

# Copy the source code into the container.
COPY . .

# Run the application.
CMD python3 main.py --mode 'ae' --dataset_dir 'save/pc-ou-dataset' --epochs 50
