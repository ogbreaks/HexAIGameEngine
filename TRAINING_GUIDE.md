# HexAI Game — Operating Guide

## Prerequisites

1. **Python 3.10+** installed (check with `python --version`).
2. **Install dependencies** (one-time setup):
   ```powershell
   pip install -r requirements-training.txt
   ```
   This creates the `.venv-training` environment with PyTorch, FastAPI, and all training dependencies.
3. **Activate the venv** before every session:
   ```powershell
   .venv-training\Scripts\Activate.ps1
   ```
   Your prompt should show `(.venv-training)` at the left. If you skip this step, commands will fail with `ModuleNotFoundError: No module named 'torch'` (or `yaml`, `numpy`, etc.).

---

## Quick Start

All commands below assume you have activated the venv and are in the project root:
```
C:\Users\origi\OneDrive\Documents\Scripts\HexAIGame\HexAIGame
```

**1. Verify everything works** (takes ~2 minutes):
```powershell
python main.py --mode train_az --config config/hex11_test.yaml
```

Expected output:
```
[Coach] Starting training on cpu
[Coach] Iteration 0/20
[Coach] Generating self-play data...
  Game 1/4 complete
  Game 2/4 complete
  Game 3/4 complete
  Game 4/4 complete
[Coach] Buffer: 4 games, ... positions
[Coach] Training step 1/10...
...
```
If you see this, the environment is working correctly.

**2. Start real training** (CPU, default config):
```powershell
python main.py --mode train_az --config config/hex11_default.yaml
```

**3. Start cloud training** (GPU, scaled config):
```powershell
python main.py --mode train_az --config config/hex11_cloud.yaml
```

---

## Modes

### 1. Train (AlphaZero)

```bash
python main.py --mode train_az --config config/<config>.yaml
```

See the [Config Files](#config-files) table for available configs.

**Resume from a checkpoint:**
```bash
python main.py --mode train_az --config config/hex11_default.yaml --resume training/checkpoints/hex_az_100.pth
```

### 2. Export Model to ONNX (for Unity)

```bash
python main.py --mode export_az --model training/models/hex_az_best.pth --output training/models/hex_az_best.onnx
```

### 3. Play Human vs Human

```bash
python main.py --mode human_vs_human
```
Enter moves as `row col` (0-indexed). Player 1 connects top↔bottom, Player 2 connects left↔right.

### 4. Random vs Random (benchmark)

```bash
python main.py --mode random_vs_random --games 1000
```

### 5. Game Server (HTTP API for Unity)

```bash
python main.py --mode server
```
Runs on `localhost:5000`. Endpoints: `/state`, `/move`, `/reset`, `/mcts_move`.

---

## Config Files

| Config | Purpose | Network | Games/iter | Iterations |
|--------|---------|---------|------------|------------|
| `hex11_micro.yaml` | Minimal 10-iteration test | 4 blocks, 64ch | 2 | 10 |
| `hex11_test.yaml` | Quick sanity check | 4 blocks, 64ch | 4 | 20 |
| `hex11_default.yaml` | Standard training | 10 blocks, 128ch | 100 | 1000 |
| `hex11_cloud.yaml` | Scaled cloud run | 20 blocks, 256ch | 800 | 5000 |

### Key Config Parameters

**Network:**
- `trunk`: `resnet` or `vit`
- `num_res_blocks`: depth of ResNet (default: 10)
- `num_channels`: width of all layers (default: 128)

**MCTS:**
- `num_simulations`: search depth per move (default: 800)
- `c_puct`: exploration constant (default: 1.4)
- `dirichlet_alpha` / `dirichlet_weight`: root noise for exploration

**Self-play:**
- `num_workers`: CPU processes (default: 4)
- `games_per_worker`: games each worker plays per iteration (default: 25)

**Training:**
- `num_iterations`: total training cycles (default: 1000)
- `train_steps_per_iter`: gradient updates per cycle (default: 100)
- `batch_size`: training batch size (default: 512)
- `lr_init`: Adam learning rate, cosine decayed to 0 (default: 0.001)
- `weight_decay`: L2 regularisation (default: 0, cloud: 0.0001)
- `buffer_size`: max replay buffer entries (default: 500K)
- `min_buffer_size`: minimum entries before training starts (default: 10K)

**GPU Inference Server** (cloud config only):
- `use_inference_server`: `true` to batch MCTS evals on GPU (default: false)
- `inference_batch_size`: states per GPU forward pass (default: 64)
- `inference_max_wait_ms`: max wait before flushing a partial batch (default: 5)

**Arena:**
- `arena_freq`: evaluate every N iterations (default: 10)
- `arena_games`: games per evaluation (default: 40, split 50/50 first-mover)
- `arena_simulations`: MCTS sims during arena (default: 200)
- `promotion_threshold`: win rate to promote challenger (default: 0.55)

**Checkpointing:**
- `checkpoint_freq`: save every N iterations (default: 50)

---

## Environment Variables

All environment variables are optional. When set, they override the corresponding value from the YAML config file (applied by `apply_env_overrides()` in `main.py`).

### Runtime variables

| Variable | Default | Purpose |
|----------|---------|--------|
| `HOURLY_RATE` | `0.0` | Cloud compute $/hr for cost tracking |
| `TIMEZONE_OFFSET` | `0` | Hours from UTC for ETA display |
| `TRAINING_CONFIG` | `config/hex11_cloud.yaml` | Config file to use |
| `GCS_BUCKET` | *(unset)* | GCS bucket for model upload (skipped if unset) |
| `MODEL_PATH` | `training/models/hex_az_best.pth` | Model for server `/mcts_move` |

### Config override variables

| Variable | Type | Config key |
|----------|------|------------|
| `NUM_ITERATIONS` | int | `num_iterations` |
| `NUM_WORKERS` | int | `num_workers` |
| `GAMES_PER_WORKER` | int | `games_per_worker` |
| `NUM_SIMULATIONS` | int | `num_simulations` |
| `BATCH_SIZE` | int | `batch_size` |
| `TRAIN_STEPS_PER_ITER` | int | `train_steps_per_iter` |
| `MIN_BUFFER_SIZE` | int | `min_buffer_size` |
| `BUFFER_SIZE` | int | `buffer_size` |
| `LR_INIT` | float | `lr_init` |
| `ARENA_FREQ` | int | `arena_freq` |
| `ARENA_GAMES` | int | `arena_games` |
| `ARENA_SIMULATIONS` | int | `arena_simulations` |
| `PROMOTION_THRESHOLD` | float | `promotion_threshold` |
| `NUM_RES_BLOCKS` | int | `num_res_blocks` |
| `NUM_CHANNELS` | int | `num_channels` |
| `CHECKPOINT_FREQ` | int | `checkpoint_freq` |
| `USE_INFERENCE_SERVER` | bool | `use_inference_server` |
| `WEIGHT_DECAY` | float | `weight_decay` |

Overridden values are logged on startup:
```
[Config] num_iterations overridden by env: 10
[Config] num_workers overridden by env: 2
```

Example (local):
```powershell
$env:HOURLY_RATE = "0.35"
$env:TIMEZONE_OFFSET = "10"
python main.py --mode train_az --config config/hex11_cloud.yaml
```

Example (Docker):
```bash
docker run -e NUM_ITERATIONS=5 -e NUM_SIMULATIONS=20 pixelpunk77/hexai-az:latest
```

---

## Monitoring

The dashboard runs in a **separate terminal** from training. Keep training running in Terminal 1 and open a new terminal (Terminal 2) for the dashboard.

**Terminal 2 — start the dashboard:**
```powershell
.venv-training\Scripts\Activate.ps1
uvicorn metrics_server:app --port 8000
```
Open `http://localhost:8000/dashboard` in your browser — auto-refreshes every 10s.

> Port 8000 is the uvicorn default. Use `--port 8080` (or any free port) if 8000 is already in use.

> **Don't run this in the same terminal as training** — it will block training. Always use a second terminal.

**Metrics API** (same port):
- `GET /metrics` — raw JSON (progress, cost, losses, ETA)
- `GET /health` — check if metrics file exists

**Metrics file:** `training/metrics.json` (written after every iteration).

---

## Output Files

| Path | Contents |
|------|----------|
| `training/models/hex_az_best.pth` | Best network weights (latest promoted) |
| `training/models/hex_az_{iter}.onnx` | ONNX export after each promotion |
| `training/checkpoints/hex_az_{iter}.pth` | Full checkpoint (weights + optimizer + scheduler) |
| `training/checkpoints/hex_az_final.pth` | Checkpoint at training completion |
| `training/metrics.json` | Current training metrics (for dashboard) |
| `training/cost_state.json` | Accumulated training hours (persists across restarts) |

---

## Training Flow Summary

Each iteration:
1. **Self-play** — Workers play MCTS games, producing (state, policy, value) training data
2. **Train** — Sample from replay buffer, update network via policy + value loss
3. **Arena** (every N iters) — Challenger vs champion; promote if win rate > 55%
4. **Checkpoint** (every M iters) — Save full training state for resume
5. **Metrics** — Write progress to JSON for dashboard

---

## Common Issues

**"No module named 'yaml'"** → You're not in the training venv. Run:
```
.venv-training\Scripts\Activate.ps1
```

**"Missing key(s) in state_dict"** → Old checkpoint was trained with a different network size. Harmless warning from the server endpoint; training itself creates a fresh network. Delete the old checkpoint to silence it:
```
del training\models\hex_az_best.pth
```

**Training hangs on "Generating self-play data" (Windows)** → Windows uses the `spawn` multiprocessing start method. Multi-worker self-play may hang or crash. Use `num_workers: 1` in your config for local Windows runs. Multi-worker works correctly on Linux (cloud).

**Training is very slow on CPU** → Expected. The default config does 800 MCTS sims × 100 games on CPU. For speed, use `hex11_test.yaml` for verification or `hex11_cloud.yaml` with GPU.

---

## Cloud Deployment (GCE + Docker)

For full cloud deployment commands, persistent disk setup, GCS model storage, and troubleshooting, see **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)**.

### Overview

1. **Build and push** the Docker image after any code change:
   ```powershell
   docker build -t pixelpunk77/hexai-az:latest .
   docker push pixelpunk77/hexai-az:latest
   ```

2. **Create a GCE instance** — the startup script (`startup-cpu.sh` or `startup-gpu.sh`) automatically installs Docker, pulls the image, and runs training.

3. **Model upload** — after training completes, `entrypoint.sh` uploads `.pth` and `.onnx` files to GCS with timestamped filenames (e.g. `hex_az_best_20260322_201524.pth`).

4. **Auto-shutdown** — a `trap on_exit EXIT` in both startup scripts guarantees the GCE instance stops itself on success or failure (30s delay for final metrics scrape).

5. **Persistent disk** (`hexai-data`) — stores Docker layer cache and models at `/mnt/hexai/`. First pull takes ~40 mins; subsequent runs are near-instant.

### Monitoring a Running Instance

Use the **GCP Console browser SSH** (no local SSH client needed):
1. Go to **Compute Engine → VM instances**
2. Click the **SSH** button next to your instance
3. A browser-based terminal opens directly on the VM

Useful commands once connected (use `sudo` — Docker dirs are root-owned):
```bash
# Check persistent disk is mounted
df -h /mnt/hexai

# Check Docker is using persistent disk
sudo docker info | grep "Docker Root Dir"
# Should show: /mnt/hexai/docker

# See Docker layer cache size
sudo du -sh /mnt/hexai/docker/

# List saved models
ls -la /mnt/hexai/models/

# Check if container is running
sudo docker ps

# Follow container logs live
sudo docker logs -f $(sudo docker ps -q)

# View startup script logs
sudo journalctl -u google-startup-scripts -f
```

### Creating a GPU Instance

Use the **Deep Learning VM** image (`common-cu121`) — it ships with NVIDIA drivers and CUDA pre-installed. The plain `ubuntu-2204-lts` image does **not** include GPU drivers and will fail with `libnvidia-ml.so.1: cannot open shared object file`.

```bash
gcloud compute instances create hexai-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  "--accelerator=type=nvidia-tesla-t4,count=1" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --scopes=cloud-platform \
  --tags=hexai-training \
  --image-family=common-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB \
  "--disk=name=hexai-data,device-name=hexai-data,auto-delete=no" \
  --metadata-from-file=startup-script=startup-gpu.sh
```

> **Why `common-cu128-ubuntu-2204-nvidia-570`?** The startup script installs the NVIDIA *container toolkit* (which wires up Docker's `--gpus` flag) but not the underlying GPU *driver*. The Deep Learning VM image provides the driver layer (`libnvidia-ml.so.1`) that the toolkit depends on.

The startup script's default env vars are set for a meaningful GPU run:

| Variable | Default | Notes |
|----------|---------|-------|
| `NUM_ITERATIONS` | `200` | Override with `--metadata` or edit script |
| `NUM_WORKERS` | `4` | Parallel self-play workers |
| `NUM_SIMULATIONS` | `400` | MCTS sims per move |

> The startup script is uploaded to GCE metadata at `gcloud create` time. Any edits to `startup-gpu.sh` take effect on the **next** `gcloud create` invocation — no separate metadata update step needed.

### Diagnosing Startup Failures

Startup script stdout/stderr is captured by Cloud Logging. Query in **Logs Explorer**:
```
resource.type="gce_instance"
resource.labels.instance_id="hexai-gpu"
logName=~"google-startup-scripts"
```

Common failures:

| Error | Cause | Fix |
|-------|-------|-----|
| `libnvidia-ml.so.1: cannot open shared object file` | Plain Ubuntu image, no GPU driver | Use `common-cu121` image |
| `exit status 127` in startup-script | Command not found (often `blkid`/`mount` from stripped PATH) | Check full log for preceding line |
| VM stops after ~2 min | Script failed before Docker pull | Check startup-script logs above |

### Key files

| File | Purpose |
|------|---------|
| `Dockerfile` | Container image (PyTorch + gcloud CLI) |
| `entrypoint.sh` | Container entrypoint: metrics server, training, GCS upload |
| `startup-cpu.sh` | GCE startup: persistent disk, Docker install, CPU training |
| `startup-gpu.sh` | GCE startup: NVIDIA toolkit, Docker install, GPU training |
| `requirements-docker.txt` | Python deps for container (includes `google-cloud-storage`) |
