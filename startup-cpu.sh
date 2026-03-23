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

# Point Docker + containerd storage to persistent disk
mkdir -p $MOUNT_POINT/docker
mkdir -p $MOUNT_POINT/containerd
mkdir -p $MOUNT_POINT/models

# Install Docker first so /etc/docker exists
curl -fsSL https://get.docker.com | sh

# Redirect Docker data-root
mkdir -p /etc/docker
cat > /etc/docker/daemon.json <<EOF
{
  "data-root": "$MOUNT_POINT/docker"
}
EOF

# Redirect containerd root (Docker 29+ stores image layers here)
systemctl stop containerd
mkdir -p /etc/containerd
containerd config default > /etc/containerd/config.toml
sed -i 's|root = "/var/lib/containerd"|root = "'$MOUNT_POINT'/containerd"|' /etc/containerd/config.toml
systemctl start containerd

# Restart Docker to pick up new data-root
systemctl restart docker
docker pull pixelpunk77/hexai-az:latest
docker run \
  -e HOURLY_RATE=${HOURLY_RATE:-0.19} \
  -e TIMEZONE_OFFSET=${TIMEZONE_OFFSET:-1} \
  -e TRAINING_CONFIG=${TRAINING_CONFIG:-config/hex11_test.yaml} \
  -e GCS_BUCKET=${GCS_BUCKET:-hexai-models} \
  -e NUM_ITERATIONS=${NUM_ITERATIONS:-1} \
  -e NUM_WORKERS=${NUM_WORKERS:-1} \
  -e NUM_SIMULATIONS=${NUM_SIMULATIONS:-5} \
  -p 8080:8080 \
  -v /mnt/hexai/models:/app/training/models \
  pixelpunk77/hexai-az:latest