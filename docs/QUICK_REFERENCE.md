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

## Docker — Build and Push

Always rebuild and push **before** creating the next cloud instance.

```powershell
docker build -t pixelpunk77/hexai-az:latest .
docker push pixelpunk77/hexai-az:latest
```

Only changed layers are pushed/pulled — heavy PyTorch layers stay cached on the persistent disk.

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
  --machine-type=e2-micro `
  --scopes=cloud-platform `
  --tags=hexai-training `
  --image-family=ubuntu-2204-lts `
  --image-project=ubuntu-os-cloud `
  --boot-disk-size=50GB `
  "--disk=name=hexai-data,device-name=hexai-data,auto-delete=no" `
  --metadata-from-file=startup-script=startup-cpu.sh
```

### Create GPU training instance (requires quota)
```powershell
gcloud compute instances create hexai-gpu-run `
  --zone=europe-west2-a `
  --machine-type=n1-standard-4 `
  --accelerator="type=nvidia-tesla-t4,count=1" `
  --provisioning-model=SPOT `
  --instance-termination-action=STOP `
  --scopes=cloud-platform `
  --tags=hexai-training `
  --image-family=ubuntu-2204-lts `
  --image-project=ubuntu-os-cloud `
  --boot-disk-size=50GB `
  "--disk=name=hexai-data,device-name=hexai-data,auto-delete=no" `
  --maintenance-policy=TERMINATE `
  --metadata-from-file=startup-script=startup-gpu.sh
```

> **Important:** `--scopes=cloud-platform` is required for GCS uploads and auto-shutdown. Without it, the service account can't write to GCS even if IAM permissions are correct (OAuth scope issue at the VM level).

### Delete and relaunch workflow
```powershell
gcloud compute instances delete hexai-cpu-test --zone=us-central1-a --quiet
# Then re-run the create command above
```

### Get external IP
```powershell
gcloud compute instances describe hexai-cpu-test `
  --zone=us-central1-a `
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
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

## Persistent Disk

| Property | Value |
|----------|-------|
| Name | `hexai-data` |
| Zone | `us-central1-a` |
| Size | 20GB |
| Type | `pd-standard` |

- First run populates Docker layer cache (~40 mins pull). All subsequent runs are near-instant.
- Models directory at `/mnt/hexai/models/` inside VM, mounted into the container via `-v` flag in docker run.
- `auto-delete=no` means the disk survives instance deletion.
- **Never delete this disk** — it holds the Docker cache and trained models.

---

## GCS Model Storage

All models upload to the bucket root with datestamped filenames:
```
gs://hexai-models/hex_az_best_20260322_201524.pth
gs://hexai-models/hex_az_best_20260322_201524.onnx
```

### List all models
```powershell
gcloud storage ls gs://hexai-models/
```

### Download a model
```powershell
# Replace TIMESTAMP with actual value from ls output
gcloud storage cp gs://hexai-models/hex_az_best_TIMESTAMP.pth ./training/models/
gcloud storage cp gs://hexai-models/hex_az_best_TIMESTAMP.onnx ./training/models/
```

---

## Training Environment Variables

These override values from the YAML config at runtime via `-e` flags:

| Variable | Default (CPU) | Default (GPU) | Description |
|----------|---------------|---------------|-------------|
| `NUM_ITERATIONS` | `1` | (from yaml) | Training iterations |
| `NUM_SIMULATIONS` | `5` | `800` | MCTS simulations per move |
| `NUM_WORKERS` | `1` | `4` | Parallel self-play workers |
| `TRAINING_CONFIG` | `config/hex11_test.yaml` | `config/hex11_default.yaml` | YAML config file |
| `GCS_BUCKET` | `hexai-models` | `hexai-models` | GCS bucket for model upload |
| `HOURLY_RATE` | `0.19` | `0.27` | $/hr for cost tracking |
| `TIMEZONE_OFFSET` | `1` | `1` | Hours from UTC |

---

## Auto-Shutdown (Trap)

- Instance auto-stops on both success and failure via `trap on_exit EXIT` in startup-cpu.sh.
- 30 second delay before shutdown allows the dashboard a final metrics scrape.
- If an instance stays RUNNING unexpectedly, manually stop it:
```powershell
gcloud compute instances stop hexai-cpu-test --zone=us-central1-a
```

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
4. Download model (see GCS section above)
5. Delete instance: gcloud compute instances delete hexai-cpu-test --zone=us-central1-a
6. Drop hex_az_best.onnx into Unity Assets/Models/
```

---

## Known Issues Resolved

| Issue | Fix |
|-------|-----|
| GCS 403 Forbidden error | Add `--scopes=cloud-platform` to instance create command |
| Instance not auto-stopping | Add `trap on_exit EXIT` to startup-cpu.sh |
| UTF-8 BOM in startup script | Save with ASCII encoding, not UTF-8 with BOM |
