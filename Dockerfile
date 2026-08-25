FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p \
    /app/dataset/raw \
    /app/dataset/train \
    /app/dataset/val \
    /app/dataset/test \
    /app/models \
    /app/logs \
    /app/notebooks \
    /app/config

EXPOSE 8888
EXPOSE 6006

CMD ["bash"]