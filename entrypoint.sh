#!/bin/bash
set -e

on_exit() {
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Script exited with code ${EXIT_CODE}"
    fi
    echo "Shutting down in 30 seconds (final metrics scrape)..."
    sleep 30
    INSTANCE=$(curl -sf -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/name 2>/dev/null || echo "")
    ZONE=$(curl -sf -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null \
        | cut -d/ -f4 || echo "")
    if [ -n "${INSTANCE}" ] && [ -n "${ZONE}" ]; then
        echo "Stopping instance ${INSTANCE} in zone ${ZONE}..."
        gcloud compute instances stop "${INSTANCE}" --zone="${ZONE}" --quiet
    else
        echo "Not running on GCE — skipping auto-shutdown"
    fi
    kill $METRICS_PID 2>/dev/null
}
trap on_exit EXIT

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

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    python -c "
from google.cloud import storage
import os

client = storage.Client()
bucket = client.bucket('${GCS_BUCKET}')
ts = '${TIMESTAMP}'

files = [
    ('/app/training/models/hex_az_best.pth', f'hex_az_best_{ts}.pth'),
    ('/app/training/models/hex_az_best.onnx', f'hex_az_best_{ts}.onnx'),
]

for local_path, gcs_name in files:
    if os.path.exists(local_path):
        bucket.blob(gcs_name).upload_from_filename(local_path)
        print(f'Uploaded to gs://${GCS_BUCKET}/{gcs_name}')
    else:
        print(f'Skipped {local_path} — not found')
"
else
    echo "GCS_BUCKET not set — skipping model upload"
fi

exit ${TRAINING_EXIT}
