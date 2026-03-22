# HexAI — Quick Reference Commands

## Local Development

### Activate environment (always first)
```powershell
cd C:\Users\origi\OneDrive\Documents\Scripts\HexAIGame\HexAIGame
.venv-training\Scripts\activate
```

### Start game server
```powershell
python main.py --mode server
```

### Start metrics dashboard
```powershell
uvicorn metrics_server:app --host 0.0.0.0 --port 8080 --reload
```
Then open: `http://localhost:8080/dashboard`

### Run training locally
```powershell
# Test run (~10 mins)
python main.py --mode train_az --config config/hex11_test.yaml

# Default run
python main.py --mode train_az --config config/hex11_default.yaml

# Resume from checkpoint
python main.py --mode train_az --config config/hex11_default.yaml --resume training/checkpoints/hex_az_100.pth
```

### Export model to ONNX
```powershell
python main.py --mode export_az --model training/models/hex_az_best.pth --output training/models/hex_az_best.onnx
```

---

## Docker

### Build and push
```powershell
docker build -t hexai-az:latest .
docker tag hexai-az:latest pixelpunk77/hexai-az:latest
docker push pixelpunk77/hexai-az:latest
```

### Test locally
```powershell
docker run -p 8080:8080 -p 5000:5000 `
  -e TRAINING_CONFIG=config/hex11_test.yaml `
  -e HOURLY_RATE=0.0 `
  -e TIMEZONE_OFFSET=1 `
  -v "${PWD}/training/models:/app/training/models" `
  -v "${PWD}/training/checkpoints:/app/training/checkpoints" `
  hexai-az:latest
```

### Clean up old images
```powershell
docker image prune
```

---

## Google Cloud — Instance Management

### Create CPU test instance
```powershell
gcloud compute instances create hexai-cpu-test `
  --zone=us-central1-a `
  --machine-type=n1-standard-4 `
  --scopes=storage-rw,cloud-platform `
  --tags=hexai-training `
  --image-family=ubuntu-2204-lts `
  --image-project=ubuntu-os-cloud `
  --boot-disk-size=50GB `
  --metadata-from-file=startup-script=startup-cpu.sh
```

### Create GPU training instance (requires quota)
```powershell
gcloud compute instances create hexai-training-run `
  --zone=europe-west2-a `
  --machine-type=n1-standard-4 `
  --accelerator="type=nvidia-tesla-t4,count=1" `
  --provisioning-model=SPOT `
  --instance-termination-action=STOP `
  --scopes=storage-rw,cloud-platform `
  --tags=hexai-training `
  --image-family=ubuntu-2204-lts `
  --image-project=ubuntu-os-cloud `
  --boot-disk-size=50GB `
  --maintenance-policy=TERMINATE `
  --metadata-from-file=startup-script=startup-gpu.sh
```

> **Important:** The `--scopes=storage-rw,cloud-platform` flag is required for GCS uploads and auto-shutdown to work. Without it, the service account can't write to GCS even if IAM permissions are correct (OAuth scope issue at the VM level).

### Get external IP
```powershell
gcloud compute instances describe hexai-training-run `
  --zone=europe-west2-a `
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

### Delete instance (stop all billing)
```powershell
gcloud compute instances delete hexai-training-run --zone=europe-west2-a
```

### List all instances
```powershell
gcloud compute instances list --project=hexai-training
```

### SSH via browser
```
https://console.cloud.google.com/compute/instances?project=hexai-training
```
Click **SSH** next to the instance name.

---

## Google Cloud Storage — Model Management

### List all training runs
```powershell
gcloud storage ls gs://hexai-models/
```

### List runs for a specific config
```powershell
gcloud storage ls gs://hexai-models/hex11_default/
```

### Download model from a specific run
```powershell
# Replace timestamp with actual run ID from ls output
gcloud storage cp `
  gs://hexai-models/hex11_default/20260322_143000/hex_az_best.pth `
  training/models/hex_az_best.pth

gcloud storage cp `
  gs://hexai-models/hex11_default/20260322_143000/hex_az_best.onnx `
  training/models/hex_az_best.onnx
```

### Download all files from a run
```powershell
gcloud storage cp -r `
  gs://hexai-models/hex11_default/20260322_143000/ `
  training/models/
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOURLY_RATE` | `0.0` | Cloud compute $/hr for cost tracking |
| `TIMEZONE_OFFSET` | `0` | Hours from UTC (GMT=0, BST=1) |
| `TRAINING_CONFIG` | `config/hex11_cloud.yaml` | Config file to use |
| `GCS_BUCKET` | `hexai-models` | GCS bucket for model upload |

---

## Monitoring

| URL | Purpose |
|-----|---------|
| `http://<ip>:8080/dashboard` | Live training dashboard |
| `http://<ip>:8080/metrics` | Raw JSON metrics |
| `http://<ip>:8080/health` | Health check |
| `https://console.cloud.google.com/monitoring/alerting?project=hexai-training` | GCP alerts |
| `https://console.cloud.google.com/storage/browser/hexai-models` | Saved models |

---

## After Training Completes

```
1. Phone notification arrives
2. Check dashboard: http://<ip>:8080/dashboard
3. List available models: gcloud storage ls gs://hexai-models/
4. Download model (see above)
5. Delete instance: gcloud compute instances delete hexai-training-run --zone=europe-west2-a
6. Drop hex_az_best.onnx into Unity Assets/Models/
```
