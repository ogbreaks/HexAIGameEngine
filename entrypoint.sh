#!/bin/bash
set -e

# Start metrics server in background
uvicorn metrics_server:app --host 0.0.0.0 --port 8080 &
METRICS_PID=$!

# Run training in foreground
python main.py --mode train_az --config ${TRAINING_CONFIG:-config/hex11_cloud.yaml}
TRAINING_EXIT=$?

# Write completion marker
echo '{"status":"complete","progress":{"percent":100}}' > training/metrics.json

# Upload model to GCS if bucket is specified
if [ -n "${GCS_BUCKET}" ]; then
    echo "Uploading models to GCS bucket: ${GCS_BUCKET}..."
    python -c "
from google.cloud import storage
import os

client = storage.Client()
bucket = client.bucket('${GCS_BUCKET}')

files = [
    '/app/training/models/hex_az_best.pth',
    '/app/training/models/hex_az_best.onnx',
]

for path in files:
    if os.path.exists(path):
        filename = os.path.basename(path)
        bucket.blob(filename).upload_from_filename(path)
        print(f'Uploaded {filename} to gs://${GCS_BUCKET}/{filename}')
    else:
        print(f'Skipped {path} — not found')
"
else
    echo "GCS_BUCKET not set — skipping model upload"
fi

# Keep metrics server alive for 10 minutes
echo "Training complete (exit ${TRAINING_EXIT}). Metrics available for 10 minutes..."
sleep 600

# Auto-terminate this GCE instance using metadata server
# Only runs inside GCE — fails gracefully elsewhere
INSTANCE_NAME=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null || echo "")
ZONE=$(curl -sf -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null \
    | cut -d/ -f4 || echo "")

if [ -n "${INSTANCE_NAME}" ] && [ -n "${ZONE}" ]; then
    echo "Shutting down instance ${INSTANCE_NAME} in zone ${ZONE}..."
    gcloud compute instances stop "${INSTANCE_NAME}" --zone="${ZONE}" --quiet
else
    echo "Not running on GCE — skipping auto-shutdown"
fi

kill $METRICS_PID 2>/dev/null
exit ${TRAINING_EXIT}
