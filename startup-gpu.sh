#!/bin/bash
curl -fsSL https://get.docker.com | sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
docker pull pixelpunk77/hexai-az:latest
docker run --gpus all \
  -e HOURLY_RATE=${HOURLY_RATE:-0.27} \
  -e TIMEZONE_OFFSET=${TIMEZONE_OFFSET:-1} \
  -e TRAINING_CONFIG=${TRAINING_CONFIG:-config/hex11_default.yaml} \
  -e GCS_BUCKET=${GCS_BUCKET:-hexai-models} \
  -p 8080:8080 \
  pixelpunk77/hexai-az:latest
