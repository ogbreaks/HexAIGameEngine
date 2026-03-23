# metrics_server.py
import json
import os

import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="HexAI Training Monitor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Resolve path relative to this file regardless of cwd
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(_BASE_DIR, "training", "metrics.json")

# NVIDIA GPU monitoring — optional, graceful fallback if driver absent
_gpu_available = False
try:
    import pynvml

    pynvml.nvmlInit()
    _gpu_available = True
except Exception:
    pass


@app.get("/metrics")
def get_metrics() -> JSONResponse:
    if not os.path.exists(METRICS_PATH):
        return JSONResponse(
            {
                "status": "waiting",
                "message": "Training has not started yet or metrics file not found.",
                "progress": {"percent": 0},
            }
        )
    try:
        with open(METRICS_PATH, "r") as f:
            data = json.load(f)
        return JSONResponse(data)
    except (json.JSONDecodeError, OSError):
        # File mid-write (should not happen with atomic os.replace, but kept as safety net)
        return JSONResponse(
            {
                "status": "reading_error",
                "message": "Metrics file temporarily unavailable.",
            }
        )


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(
        """
<!DOCTYPE html>
<html>
<head>
    <title>HexAI Training Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f0f0f; color: #e0e0e0; padding: 20px;
        }
        h1 { color: #fff; font-size: 1.4em; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
        .card {
            background: #1a1a1a; border-radius: 12px; padding: 20px;
            border: 1px solid #2a2a2a;
        }
        .card h2 { font-size: 0.75em; text-transform: uppercase;
                   letter-spacing: 0.1em; color: #666; margin-bottom: 14px; }
        .metric { display: flex; justify-content: space-between;
                  align-items: baseline; margin-bottom: 10px; }
        .metric .label { color: #888; font-size: 0.85em; }
        .metric .value { color: #fff; font-size: 1.0em; font-weight: 500; }
        .metric .value.highlight { color: #4ade80; font-size: 1.1em; }
        .progress-bar-bg {
            background: #2a2a2a; border-radius: 8px; height: 12px;
            margin: 8px 0 16px 0; overflow: hidden;
        }
        .progress-bar-fill {
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            height: 100%; border-radius: 8px;
            transition: width 0.5s ease;
        }
        .status-pill {
            display: inline-block; padding: 4px 12px; border-radius: 20px;
            font-size: 0.8em; font-weight: 600; margin-bottom: 16px;
        }
        .status-training { background: #1d4ed8; color: #bfdbfe; }
        .status-complete { background: #166534; color: #bbf7d0; }
        .status-waiting  { background: #4b4b4b; color: #ccc; }
        .footer { margin-top: 20px; color: #444; font-size: 0.75em; text-align: center; }
        .eta-big { font-size: 1.3em; color: #4ade80; font-weight: 600; }
        /* Hardware bars */
        .bar-section { margin-top: 10px; }
        .bar-section label { display:block; color:#888; font-size:0.75em; margin-bottom:6px; }
        .bar-row { display:flex; align-items:flex-end; gap:3px; height:40px; }
        .cpu-bar {
            flex:1; background:#3b82f6; border-radius:3px 3px 0 0;
            transition: height 0.4s ease; min-height:2px;
        }
        .gpu-bar-wrap { display:flex; align-items:flex-end; height:40px; width:100%; }
        .gpu-bar {
            background: linear-gradient(180deg,#f97316,#ef4444);
            border-radius:3px 3px 0 0; width:100%;
            transition: height 0.4s ease; min-height:2px;
        }
        .bar-floor { border-top:1px solid #333; margin-top:2px; }
        .hw-metric { display:flex; justify-content:space-between;
                     font-size:0.8em; color:#888; margin-top:6px; }
        .hw-metric span { color:#ccc; }
        /* Phase badge */
        .phase-badge {
            display:inline-block; padding:3px 10px; border-radius:12px;
            font-size:0.75em; font-weight:600; letter-spacing:0.05em;
            text-transform:uppercase;
        }
        .phase-self_play  { background:#7c3aed; color:#e0d4fc; }
        .phase-training   { background:#0d9488; color:#ccfbf1; }
        .phase-arena      { background:#b45309; color:#fef3c7; }
        .phase-checkpoint  { background:#4338ca; color:#c7d2fe; }
        .phase-idle       { background:#333;    color:#888; }
        .phase-init       { background:#333;    color:#888; }
        /* Sub-progress mini bar */
        .sub-progress-wrap {
            margin-top:8px; display:flex; align-items:center; gap:10px;
        }
        .sub-bar-bg {
            flex:1; background:#2a2a2a; border-radius:6px; height:6px;
            overflow:hidden;
        }
        .sub-bar-fill {
            background:linear-gradient(90deg,#6366f1,#a78bfa);
            height:100%; border-radius:6px;
            transition:width 0.5s ease;
        }
        .sub-label { color:#888; font-size:0.8em; white-space:nowrap; }
    </style>
</head>
<body>
    <h1>&#x2B21; HexAI Training Monitor</h1>
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
        <div id="status-pill" class="status-pill status-waiting">Waiting</div>
        <div id="phase-badge" class="phase-badge phase-init" style="display:none;"></div>
        <span id="loading-msg" style="color:#666; font-size:0.85em;">Loading data...</span>
    </div>
    <div class="grid">

        <div class="card">
            <h2>Progress</h2>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" id="progress-fill" style="width:0%"></div>
            </div>
            <div class="metric">
                <span class="label">Complete</span>
                <span class="value highlight" id="percent">&#8212;</span>
            </div>
            <div class="metric">
                <span class="label">Steps done</span>
                <span class="value" id="steps-done">&#8212;</span>
            </div>
            <div class="metric">
                <span class="label">Steps total</span>
                <span class="value" id="steps-total">&#8212;</span>
            </div>
            <div class="metric">
                <span class="label">Speed</span>
                <span class="value" id="fps">&#8212;</span>
            </div>
            <div id="sub-progress-row" class="sub-progress-wrap" style="display:none;">
                <div class="sub-bar-bg">
                    <div class="sub-bar-fill" id="sub-fill" style="width:0%"></div>
                </div>
                <span class="sub-label" id="sub-label">&#8212;</span>
            </div>
        </div>

        <div class="card">
            <h2>Time</h2>
            <div class="metric">
                <span class="label">Elapsed (session)</span>
                <span class="value" id="elapsed-session">&#8212;</span>
            </div>
            <div class="metric">
                <span class="label">Elapsed (total)</span>
                <span class="value" id="elapsed-total">&#8212;</span>
            </div>
            <div class="metric">
                <span class="label">Time remaining</span>
                <span class="value" id="eta-hours">&#8212;</span>
            </div>
            <div class="metric">
                <span class="label">ETA</span>
                <span class="value eta-big" id="eta-label">&#8212;</span>
            </div>
        </div>

        <div class="card">
            <h2>Cost</h2>
            <div class="metric">
                <span class="label">Rate</span>
                <span class="value" id="rate">&#8212;</span>
            </div>
            <div class="metric">
                <span class="label">Spent so far</span>
                <span class="value highlight" id="cost-so-far">&#8212;</span>
            </div>
            <div class="metric">
                <span class="label">Estimated total</span>
                <span class="value" id="cost-total">&#8212;</span>
            </div>
        </div>

        <div class="card">
            <h2>Training Quality</h2>
            <div class="metric">
                <span class="label">Win rate (last 200)</span>
                <span class="value highlight" id="win-rate">&#8212;</span>
            </div>
            <div class="metric">
                <span class="label">Entropy loss</span>
                <span class="value" id="entropy">&#8212;</span>
            </div>
            <div class="metric">
                <span class="label">Value loss</span>
                <span class="value" id="value-loss">&#8212;</span>
            </div>
            <div class="metric">
                <span class="label">Policy loss</span>
                <span class="value" id="policy-loss">&#8212;</span>
            </div>
            <div class="metric">
                <span class="label">Opponent upgrades</span>
                <span class="value" id="upgrades">&#8212;</span>
            </div>
        </div>

        <div class="card">
            <h2>Hardware</h2>
            <div class="bar-section">
                <label>CPU Cores</label>
                <div class="bar-row" id="cpu-bars"></div>
                <div class="bar-floor"></div>
            </div>
            <div class="bar-section" id="gpu-section" style="display:none; margin-top:14px;">
                <label>GPU Utilisation</label>
                <div class="gpu-bar-wrap">
                    <div class="gpu-bar" id="gpu-bar" style="height:2px"></div>
                </div>
                <div class="bar-floor"></div>
                <div class="hw-metric">
                    <span>Util</span><span id="gpu-util">&#8212;</span>
                </div>
                <div class="hw-metric">
                    <span>VRAM</span><span id="gpu-vram">&#8212;</span>
                </div>
                <div class="hw-metric">
                    <span>Temp</span><span id="gpu-temp">&#8212;</span>
                </div>
            </div>
        </div>

    </div>

<script>
async function refreshHardware() {
    try {
        const r = await fetch('/hardware');
        const d = await r.json();

        // CPU bars
        const cpuBars = document.getElementById('cpu-bars');
        if (cpuBars.children.length !== d.cpu.length) {
            cpuBars.innerHTML = d.cpu.map(() => '<div class="cpu-bar"></div>').join('');
        }
        d.cpu.forEach((pct, i) => {
            cpuBars.children[i].style.height = Math.max(2, pct * 0.4) + 'px';
            cpuBars.children[i].title = pct.toFixed(0) + '%';
        });

        // GPU bars
        const gpuSection = document.getElementById('gpu-section');
        if (d.gpu) {
            gpuSection.style.display = '';
            document.getElementById('gpu-bar').style.height = Math.max(2, d.gpu.util * 0.4) + 'px';
            document.getElementById('gpu-util').textContent = d.gpu.util + '%';
            document.getElementById('gpu-vram').textContent =
                d.gpu.mem_used_gb + ' / ' + d.gpu.mem_total_gb + ' GB';
            document.getElementById('gpu-temp').textContent = d.gpu.temp + '\u00b0C';
        } else {
            gpuSection.style.display = 'none';
        }
    } catch(e) {}
}

refreshHardware();
setInterval(refreshHardware, 2000);
</script>

<script>
async function refresh() {
    try {
        const r = await fetch('/metrics');
        const d = await r.json();

        // Status pill
        const pill = document.getElementById('status-pill');
        pill.textContent = d.status.charAt(0).toUpperCase() + d.status.slice(1);
        pill.className = 'status-pill status-' + d.status;

        // Loading message
        const msg = document.getElementById('loading-msg');
        const hasPhase = d.phase && d.phase.name && d.phase.name !== 'init';
        if (d.status === 'training' && (d.progress?.fps ?? 0) === 0 && !hasPhase) {
            msg.textContent = 'Loading data...';
            msg.style.color = '#666';
        } else if (d.status === 'complete') {
            msg.textContent = 'Training complete.';
            msg.style.color = '#4ade80';
        } else if (d.status === 'training') {
            msg.textContent = '';
        } else {
            msg.textContent = 'Loading data...';
            msg.style.color = '#666';
        }

        // Progress
        const pct = d.progress?.percent ?? 0;
        document.getElementById('progress-fill').style.width = pct + '%';
        document.getElementById('percent').textContent = pct + '%';
        document.getElementById('steps-done').textContent =
            (d.progress?.steps_completed ?? 0).toLocaleString();
        document.getElementById('steps-total').textContent =
            (d.progress?.steps_total ?? 0).toLocaleString();
        document.getElementById('fps').textContent =
            d.progress?.fps != null ? d.progress.fps + ' steps/s' : '\u2014';

        // Phase badge
        const phase = d.phase;
        const badge = document.getElementById('phase-badge');
        if (phase && phase.name && d.status === 'training') {
            const labels = {self_play:'Self-Play', training:'Training', arena:'Arena', checkpoint:'Checkpoint', idle:'Idle', init:'Init'};
            badge.textContent = labels[phase.name] ?? phase.name;
            badge.className = 'phase-badge phase-' + phase.name;
            badge.style.display = '';
        } else {
            badge.style.display = 'none';
        }

        // Sub-progress
        const subRow = document.getElementById('sub-progress-row');
        if (phase && phase.sub_total > 0 && d.status === 'training') {
            const subPct = Math.min(100, Math.round(phase.sub_done / phase.sub_total * 100));
            document.getElementById('sub-fill').style.width = subPct + '%';
            document.getElementById('sub-label').textContent = phase.sub_label || '';
            subRow.style.display = '';
        } else {
            subRow.style.display = 'none';
        }

        // Time
        document.getElementById('elapsed-session').textContent =
            d.time?.elapsed_session_hours != null
            ? d.time.elapsed_session_hours + ' hrs' : '\u2014';
        document.getElementById('elapsed-total').textContent =
            d.time?.elapsed_total_hours != null
            ? d.time.elapsed_total_hours + ' hrs' : '\u2014';
        document.getElementById('eta-hours').textContent =
            d.time?.eta_hours != null ? d.time.eta_hours + ' hrs' : '\u2014';
        document.getElementById('eta-label').textContent =
            d.time?.eta_label ?? '\u2014';

        // Cost
        const rate = d.cost?.hourly_rate ?? 0;
        document.getElementById('rate').textContent =
            rate > 0 ? '$' + rate.toFixed(2) + '/hr' : 'Local (free)';
        document.getElementById('cost-so-far').textContent =
            d.cost?.cost_so_far != null ? '$' + d.cost.cost_so_far.toFixed(4) : '\u2014';
        document.getElementById('cost-total').textContent =
            d.cost?.cost_estimate_total != null
            ? '$' + d.cost.cost_estimate_total.toFixed(4) : '\u2014';

        // Training quality
        const wr = d.training?.win_rate_last_200;
        document.getElementById('win-rate').textContent =
            wr != null ? (wr * 100).toFixed(1) + '%' : '\u2014';
        document.getElementById('entropy').textContent =
            d.training?.entropy_loss ?? '\u2014';
        document.getElementById('value-loss').textContent =
            d.training?.value_loss ?? '\u2014';
        document.getElementById('policy-loss').textContent =
            d.training?.policy_loss ?? '\u2014';
        document.getElementById('upgrades').textContent =
            d.training?.opponent_upgrades ?? '\u2014';

        // Footer
        document.getElementById('footer').textContent =
            'Last updated: ' + new Date().toLocaleTimeString()
            + ' \u00b7 Refreshing every 10s';

    } catch(e) {
        document.getElementById('footer').textContent = 'Connection error \u2014 retrying...';
    }
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""
    )


@app.get("/hardware")
def hardware() -> JSONResponse:
    cpu = psutil.cpu_percent(percpu=True, interval=0.1)
    result: dict = {"cpu": cpu, "gpu": None}
    if _gpu_available:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            result["gpu"] = {
                "util": util.gpu,
                "mem_used_gb": round(mem.used / 1e9, 1),
                "mem_total_gb": round(mem.total / 1e9, 1),
                "temp": temp,
            }
        except Exception:
            pass
    return JSONResponse(result)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {"ok": True, "metrics_file_exists": os.path.exists(METRICS_PATH)}
    )


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": "HexAI Training Monitor",
            "endpoints": ["/dashboard", "/metrics", "/health"],
        }
    )
