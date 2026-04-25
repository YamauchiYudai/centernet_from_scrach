FROM python:3.10-slim

# tzdataやOpenCVのビルド等に必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先に依存パッケージをインストール（キャッシュ活用のため）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 後はdocker-composeでバインドマウントするため、COPYは必須ではないが記載
COPY . /app

CMD ["/bin/bash"]
