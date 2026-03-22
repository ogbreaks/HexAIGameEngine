#!/bin/bash
curl -fsSL https://get.docker.com | sh
systemctl restart docker
docker pull pixelpunk77/hexai-az:latest
docker run \
  -e HOURLY_RATE=${HOURLY_RATE:-0.19} \
  -e TIMEZONE_OFFSET=${TIMEZONE_OFFSET:-1} \
  -e TRAINING_CONFIG=${TRAINING_CONFIG:-config/hex11_test.yaml} \
  -e GCS_BUCKET=${GCS_BUCKET:-hexai-models} \
  -p 8080:8080 \
  pixelpunk77/hexai-az:latest
