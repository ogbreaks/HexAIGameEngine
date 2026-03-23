#!/bin/bash

INSTANCE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
ZONE=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | cut -d/ -f4)

on_exit() {
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ENTRYPOINT FAILED with exit code $EXIT_CODE — shutting down"
    else
        echo "ENTRYPOINT SUCCESS — shutting down"
    fi
    sleep 30
    gcloud compute instances stop "$INSTANCE" --zone="$ZONE" --quiet
}

trap on_exit EXIT


# Mount persistent disk
DISK_DEV="/dev/disk/by-id/google-hexai-data"
MOUNT_POINT="/mnt/hexai"

mkdir -p $MOUNT_POINT

# Format only if not already formatted
if ! blkid $DISK_DEV; then
    mkfs.ext4 -F $DISK_DEV
fi

mount $DISK_DEV $MOUNT_POINT

# Point Docker storage to persistent disk
mkdir -p $MOUNT_POINT/docker
mkdir -p $MOUNT_POINT/models

# Install Docker first so /etc/docker exists
curl -fsSL https://get.docker.com | sh

# Install NVIDIA container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker

# NOW write daemon.json - directory exists
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<EOF
{
  "data-root": "$MOUNT_POINT/docker"
}
EOF

# Restart Docker to pick up new data-root + NVIDIA runtime
systemctl restart docker

# Now pull - layers go to persistent disk
docker pull pixelpunk77/hexai-az:latest
docker run --gpus all \
  -e HOURLY_RATE=${HOURLY_RATE:-0.27} \
  -e TIMEZONE_OFFSET=${TIMEZONE_OFFSET:-1} \
  -e TRAINING_CONFIG=${TRAINING_CONFIG:-config/hex11_default.yaml} \
  -e GCS_BUCKET=${GCS_BUCKET:-hexai-models} \
  -e NUM_ITERATIONS=${NUM_ITERATIONS:-1} \
  -e NUM_WORKERS=${NUM_WORKERS:-1} \
  -e NUM_SIMULATIONS=${NUM_SIMULATIONS:-5} \
  -p 8080:8080 \
  -v /mnt/hexai/models:/app/training/models \
  pixelpunk77/hexai-az:latest
