#!/bin/bash
set -e

# Start metrics server in background
uvicorn metrics_server:app --host 0.0.0.0 --port 8080 &
METRICS_PID=$!

# Run training in foreground (not exec — we need post-training steps)
python main.py --mode train_az --config ${TRAINING_CONFIG:-config/hex11_cloud.yaml}
TRAINING_EXIT=$?

# Write completion marker
echo '{"status":"complete","progress":{"percent":100}}' > training/metrics.json

# Keep metrics server alive for 10 minutes so dashboard stays readable
echo "Training complete (exit $TRAINING_EXIT). Metrics available for 10 minutes..."
sleep 600

# Clean shutdown
kill $METRICS_PID
exit $TRAINING_EXIT
