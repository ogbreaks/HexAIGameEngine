# HexAIGameEngine — Technical Handoff Report

Generated: 2026-03-18  
Commit: `45f47e8` (main)  
Repository: https://github.com/ogbreaks/HexAIGameEngine

---

## 1. Directory Tree

```
HexAIGame/
├── main.py                          # CLI entry point (human_vs_human / server modes)
├── game/
│   ├── __init__.py
│   ├── hex_game.py                  # Immutable HexGame state + rules
│   ├── hex_actions.py               # Legal move enumeration
│   ├── hex_state_vector.py          # Canonical 122-float state vector
│   └── tests/
│       ├── test_hex_game.py
│       ├── test_hex_actions.py
│       └── test_hex_state_vector.py
├── server/
│   ├── __init__.py
│   ├── hex_server.py                # HTTP server (stdlib, no framework)
│   ├── game_manager.py              # Single-game state container
│   └── tests/
│       ├── test_hex_server.py
│       └── test_game_manager.py
├── training/
│   ├── __init__.py
│   ├── hex_env.py                   # Gymnasium environment (self-play)
│   ├── hex_train.py                 # PPO training script
│   ├── hex_mcts.py                  # Monte Carlo Tree Search
│   ├── hex_export.py                # ONNX export for Unity Sentis
│   ├── HexTraining.ipynb            # Jupyter notebook interface
│   ├── models/
│   │   ├── hex_easy.zip             # Trained SB3 PPO model (easy)
│   │   └── hex_easy.onnx            # Exported ONNX (easy)
│   ├── checkpoints/
│   │   ├── easy/                    # (empty — training complete)
│   │   └── hard/
│   │       ├── hex_hard_50000_steps.zip
│   │       ├── hex_hard_100000_steps.zip
│   │       ├── hex_hard_150000_steps.zip
│   │       ├── hex_hard_1300000_steps.zip
│   │       ├── hex_hard_1350000_steps.zip
│   │       └── hex_hard_1400000_steps.zip
│   └── results/
│       ├── easy/PPO_1 … PPO_11/     # TensorBoard event files
│       └── hard/PPO_1 … PPO_4/
└── Assets/Scripts/
    ├── AIManager.cs                 # Local ONNX inference (Unity Sentis fallback)
    ├── DifficultySelector.cs        # Difficulty UI + sim-count wiring
    ├── GameStateData.cs             # JSON data classes
    ├── HexBoard.cs                  # Central board controller
    ├── HexCell.cs                   # Per-cell visual + state
    ├── HexClient.cs                 # HTTP client (UnityWebRequest)
    ├── HexInputHandler.cs           # Input routing
    ├── UIManager.cs                 # UI display (turn, winner, difficulty panel)
    └── Editor/
        ├── HexBoardSetup.cs         # Editor menu item to build board scene
        └── ARCHITECTURE.md
```

---

## 2. Python Source Files

### `game/hex_game.py`
**Purpose:** Immutable `HexGame` snapshot with board state, move application, and win detection via BFS.

```python
"""
hex_game.py — immutable Hex game state and rules.

Board layout
------------
11 × 11 grid stored as a flat list of 121 ints.
  0 = empty   1 = Player 1   2 = Player 2

Win conditions
--------------
Player 1 connects Row 0 (top) to Row 10 (bottom).
Player 2 connects Col 0 (left) to Col 10 (right).

Hex-grid adjacency (pointy-top, row-major indexing)
----------------------------------------------------
Neighbours of (r, c):
  (r-1, c)   upper-left
  (r-1, c+1) upper-right
  (r,   c-1) left
  (r,   c+1) right
  (r+1, c-1) lower-left
  (r+1, c)   lower-right
"""

from __future__ import annotations
from collections import deque

SIZE = 11

_DIRECTIONS: list[tuple[int, int]] = [
    (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0),
]

class HexGame:
    """Immutable snapshot of a Hex game."""

    def __init__(self) -> None:
        self._board: list[int] = [0] * (SIZE * SIZE)
        self._current_player: int = 1
        self._terminal: bool = False
        self._winner: int | None = None

    def get_board(self) -> list[int]: ...          # copy of flat board
    def get_current_player(self) -> int: ...       # 1 or 2
    def is_terminal(self) -> bool: ...
    def get_winner(self) -> int | None: ...        # 1, 2, or None
    def apply_action(self, action: int) -> HexGame: ...  # returns NEW state
    def print_board(self) -> None: ...
```

---

### `game/hex_actions.py`
**Purpose:** Enumerate legal moves (empty cell indices) and convert between flat index and (row, col).

```python
SIZE = 11  # imported from hex_game

def get_legal_actions(game_state: HexGame) -> list[int]:
    # Returns [] if terminal; otherwise indices of empty cells in row-major order

def action_to_rowcol(action: int) -> tuple[int, int]:
    return (action // SIZE, action % SIZE)

def rowcol_to_action(row: int, col: int) -> int:
    return row * SIZE + col
```

---

### `game/hex_state_vector.py`
**Purpose:** Produces the canonical 122-float observation vector shared by training, server, and Unity.

```python
STATE_VECTOR_SIZE: int = 122

# Encoding:
#   board[i] == 0  →  0.0   (empty)
#   board[i] == 1  →  1.0   (Player 1)
#   board[i] == 2  → -1.0   (Player 2)
#   [121]          →  1.0 if P1 to move, -1.0 if P2

def get_state_vector(game_state: HexGame) -> list[float]: ...
```

---

### `server/game_manager.py`
**Purpose:** Thread-safe single-game container; translates `HexGame` into JSON-serialisable dicts for the HTTP layer.

Key methods:
- `get_state() -> dict` — returns full state dict
- `apply_move(action: int) -> dict` — validates action, applies it, returns new state dict
- `reset() -> dict` — new game, returns initial state dict
- `get_mcts_action(mcts, simulations=None) -> dict` — runs MCTS on current snapshot (does **not** mutate game)

State dict shape:
```json
{
  "state_vector":   [<122 floats>],
  "legal_actions":  [<0–121 ints>],
  "current_player": 1,
  "is_terminal":    false,
  "winner":         null
}
```

---

### `server/hex_server.py`
**Purpose:** Stdlib-only threaded HTTP server (no Flask/FastAPI) at `localhost:5000`. Loads PPO model at startup for MCTS.

See Section 7 for all endpoint details.

Model loading:
```python
_MODEL_PATH = os.environ.get("MODEL_PATH",
    "<repo>/training/models/hex_easy.zip")
# Loaded with:
PPO.load(path, custom_objects={"features_extractor_class": HexCNNExtractor})
```

---

### `training/hex_env.py`
**Purpose:** Gymnasium environment wrapping `HexGame` for SB3 PPO training with self-play opponent.

```python
observation_space = Box(-1.0, 1.0, shape=(122,), dtype=float32)
action_space      = Discrete(121)

# Rewards: +1.0 agent wins, -1.0 agent loses, 0.0 game continues
# Illegal action masking: substitutes a random legal action silently
# Opponent: random (Phase 1) → PPO model (Phase 2 via set_opponent_model())
```

---

### `training/hex_train.py`
**Purpose:** PPO training script with residual CNN feature extractor and self-play callback.

**Architecture — `HexCNNExtractor` (current):**
```
Input: [B, 122]
  board  → [B, 1, 11, 11]
         → Conv2d(1→64, 3×3, pad=1) → BN → ReLU          (stem)
         → 6 × ResidualBlock(64)                           (residual tower)
         → AdaptiveAvgPool2d(1) → flatten → [B, 64]        (global avg pool)
  player → [B, 1]
  concat → [B, 65]
         → Linear(65→256) → ReLU
         → Linear(256→128)                                 (features_dim=128)
Output: [B, 128]
```

**`ResidualBlock`:**
```python
def forward(self, x):
    residual = x
    x = F.relu(self.bn1(self.conv1(x)))   # Conv → BN → ReLU
    x = self.bn2(self.conv2(x))            # Conv → BN
    return F.relu(x + residual)            # add + ReLU
```

**PPO hyperparameters:**
```python
n_steps      = 2048
batch_size   = 64
n_epochs     = 10
learning_rate = 3e-4
gamma        = 0.99
gae_lambda   = 0.95
clip_range   = 0.2
ent_coef     = 0.01
```

**Difficulty levels:**
| Level  | total_steps | swap_freq  |
|--------|-------------|------------|
| easy   | 1,500,000   | 75,000     |
| medium | 3,000,000   | 100,000    |
| hard   | 8,000,000   | 200,000    |
| expert | 15,000,000  | 300,000    |

`CHECKPOINT_FREQ = 50,000` steps.

Self-play: `SelfPlayCallback` loads the latest checkpoint into the opponent model every `swap_freq` steps.

---

### `training/hex_mcts.py`
**Purpose:** PUCT-based MCTS using the PPO policy's action probabilities as priors and value head for position evaluation.

```python
class MCTSNode:
    state: HexGame
    parent: MCTSNode | None
    action: int | None       # action taken from parent → this node
    children: dict[int, MCTSNode]
    visits: int
    value: float             # sum of values from THIS node's player's perspective
    prior: float
    q: float                 # property: value/visits

class MCTS:
    def __init__(self, model, num_simulations=200, c_puct=1.4, add_noise=False):
        ...
    def get_action(self, game) -> int: ...
    def get_action_with_stats(self, game, simulations=None) -> (int, dict, float): ...
```

**PUCT formula:**
```
UCB = Q(s,a) + c_puct · P(s,a) · √max(N(s),1) / (1 + N(s,a))
```
- Q is negated at parent level (child stores opponent-perspective value)
- P(s,a) = softmax action probabilities from `model.policy.get_distribution()`
- V(s) = `model.policy.predict_values()`, always in player-1's frame; negated for player 2 nodes

**Dirichlet noise (root only, training only):**
```python
# add_noise=True enables this:
noise = np.random.dirichlet([0.3] * len(children))
child.prior = 0.75 * child.prior + 0.25 * noise[i]
```

**Server** instantiates `MCTS(model, add_noise=False)`.  
**Training** (if used) should instantiate with `add_noise=True`.

---

### `training/hex_export.py`
**Purpose:** Exports the PPO actor network to ONNX for consumption by Unity Sentis (Inference Engine).

---

## 3. C# Source Files (Assets/Scripts/)

### `GameStateData.cs`
**Purpose:** JSON-serialisable data classes for all server ↔ Unity communication.

```csharp
[Serializable] class GameStateData {
    float[] state_vector;   // 122 floats
    int[]   legal_actions;  // flat row-major indices of empty cells
    int     current_player; // 1 or 2
    bool    is_terminal;
    int     winner;         // 0 = none, 1 = P1, 2 = P2
}

[Serializable] class MoveRequest    { int    action; }
[Serializable] class MctsRequest    { int    simulations; }
[Serializable] class MctsResponse   { int    action; float time_ms; }
[Serializable] class ErrorResponse  { string error; int[] legal_actions; }
```

---

### `HexClient.cs`
**Purpose:** All HTTP communication via `UnityWebRequest`; every method is a coroutine.

```csharp
const string BASE_URL = "http://localhost:5000";

IEnumerator GetState(Action<GameStateData> onSuccess, Action<string> onError)
    // GET /state

IEnumerator PostMove(int action, Action<GameStateData> onSuccess, Action<string> onError)
    // POST /move  body: {"action": N}

IEnumerator PostReset(Action<GameStateData> onSuccess, Action<string> onError)
    // POST /reset

IEnumerator PostMctsMove(int simulations, Action<int> onSuccess, Action<string> onError)
    // POST /mcts_move  body: {"simulations": N}
    // onSuccess receives the action int; onError fires on network failure or non-200
```

---

### `HexBoard.cs`
**Purpose:** Central board controller; routes input, coordinates server calls, renders state, triggers AI turns.

Key fields:
```csharp
public HexClient  hexClient;
public UIManager  uiManager;
public AIManager  aiManager;          // optional local ONNX fallback
public bool       humanIsPlayer1 = true;
private int       _mctsSimulations = 100;  // set by DifficultySelector
private const int GridSize = 11;
```

AI turn flow (`TakeAITurnWithDelay`):
1. Wait 0.5 s
2. `PostMctsMove(_mctsSimulations)` → server MCTS
3. On failure → `aiManager.GetBestAction()` OR random legal move
4. `PostMove(action)` → apply on server

---

### `DifficultySelector.cs`
**Purpose:** Difficulty UI; maps buttons to simulation counts, loads ONNX model, resets game.

Simulation counts:
| Difficulty | Simulations |
|------------|-------------|
| Easy       | 8           |
| Medium     | 100         |
| Hard       | 400         |
| Expert     | 1000        |

On select: `hexBoard.SetMctsSimulations(N)` → `hexBoard.ResetGame()`.

---

### `AIManager.cs`
**Purpose:** Local ONNX inference fallback using Unity Sentis (Inference Engine) when the Python server's MCTS is unavailable.

```csharp
public enum Difficulty { Easy, Medium, Hard, Expert }

void  LoadModel(Difficulty difficulty)    // loads matching ModelAsset
int   GetBestAction(float[] stateVector, List<int> legalActions)
      // Input tensor: shape [1, 122], name "obs_0"
      // Output tensor: shape [1, 121], name (configurable, default "action_probs")
      // Returns argmax over legal actions; random fallback if no model loaded
```

---

### `HexCell.cs`
**Purpose:** Per-cell component; manages visual colour state (empty / P1 / P2 / hover / legal).

```csharp
public int row, col, action;  // action = row * 11 + col
public enum CellState { Empty, Player1, Player2 }
```
Colour scheme: P1=deep red, P2=deep blue, hover=yellow, legal=faint green.

---

### `HexInputHandler.cs`
**Purpose:** Input routing only — forwards cell clicks to `HexBoard` when `inputEnabled`.

---

### `UIManager.cs`
**Purpose:** All UI display: turn text, status messages, win announcement, reset button, difficulty panel.

---

## 4. Dependencies

### Python training venv (`.venv-training`)

| Package              | Version  |
|----------------------|----------|
| stable_baselines3    | 2.7.1    |
| torch                | 2.10.0   |
| gymnasium            | 1.2.3    |
| numpy                | 2.4.3    |
| onnx                 | 1.20.1   |
| onnxscript           | 0.6.2    |
| tensorboard          | 2.20.0   |
| matplotlib           | 3.10.8   |
| pandas               | 3.0.1    |
| psutil               | 7.2.2    |
| tqdm                 | 4.67.3   |
| cloudpickle          | 3.1.2    |
| pygame               | 2.6.1    |

Full freeze (selected):
```
stable_baselines3==2.7.1
torch==2.10.0
gymnasium==1.2.3
numpy==2.4.3
onnx==1.20.1
onnxscript==0.6.2
tensorboard==2.20.0
```

### Server runtime
No third-party packages required — uses Python stdlib only (`http.server`, `json`, `threading`, `socketserver`). SB3/PyTorch are only imported if `MODEL_PATH` resolves to a valid file.

---

## 5. State Vector Contract

**DO NOT change without updating all consumers** (training, server, Unity `AIManager.cs`).

```
Length: 122 float32 values

[0 – 120]  Board cells, row-major order (row=i//11, col=i%11):
              +1.0  = Player 1 piece
              -1.0  = Player 2 piece
               0.0  = empty

[121]      Current player to move:
              +1.0  = Player 1
              -1.0  = Player 2
```

Constants in code:
- Python: `game/hex_state_vector.py` → `STATE_VECTOR_SIZE = 122`
- Python: `game/hex_game.py` → `SIZE = 11`
- C#: `HexBoard.cs` → `const int GridSize = 11`
- C#: `AIManager.cs` → input tensor shape `[1, 122]`

---

## 6. ONNX Export

### Call signature (`training/hex_export.py`)

```python
torch.onnx.export(
    wrapper,           # _ActorWrapper(policy): obs → action_logits
    dummy_obs,         # torch.zeros(1, 122, dtype=torch.float32)
    onnx_path,
    opset_version=17,
    dynamo=False,
    export_params=True,
    input_names=["obs_0"],
    output_names=["action_probs"],
    dynamic_axes={
        "obs_0":        {0: "batch"},
        "action_probs": {0: "batch"},
    },
)
```

### ONNX tensor contract

| Tensor         | Name           | Shape     | dtype   | Notes                            |
|----------------|----------------|-----------|---------|----------------------------------|
| Input          | `obs_0`        | `[1,122]` | float32 | State vector (see §5)            |
| Output         | `action_probs` | `[1,121]` | float32 | Raw logits; argmax over legal set|

### `_ActorWrapper` (what gets exported)

Only the **actor** path is exported (not the value head):
```python
features = policy.pi_features_extractor(obs)   # HexCNNExtractor
latent_pi = policy.mlp_extractor.policy_net(features)
return policy.action_net(latent_pi)             # linear → [B, 121]
```

Unity C# reads the output tensor as logits and takes the argmax over `legal_actions`.  
The output layer name is configurable via `AIManager.outputLayerName` (default: `"action_probs"`).

---

## 7. HTTP Server Endpoints

**Base URL:** `http://localhost:5000`  
**Protocol:** HTTP/1.0, JSON bodies, CORS headers on all responses.  
**CORS:** `Access-Control-Allow-Origin: *`

---

### `GET /state`
Returns current game state.

**Response 200:**
```json
{
  "state_vector":   [<122 floats>],
  "legal_actions":  [0, 1, 2, ...],
  "current_player": 1,
  "is_terminal":    false,
  "winner":         null
}
```

---

### `POST /move`
Apply a human or AI move.

**Request:**
```json
{ "action": 60 }
```

**Response 200:** same shape as `GET /state`

**Response 400 (illegal move):**
```json
{ "error": "Illegal move", "legal_actions": [0, 1, ...] }
```

**Response 400 (bad input):**
```json
{ "error": "Missing 'action' field in request body." }
```

---

### `POST /reset`
Start a new game.

**Request:** empty body `{}`

**Response 200:** same shape as `GET /state` (fresh empty board)

---

### `POST /mcts_move`
Run MCTS and return the recommended action. Does **not** apply the move — caller must follow up with `POST /move`.

**Request:**
```json
{ "simulations": 200 }
```
`simulations` is optional; omitting it uses the server default (`num_simulations=200`).

**Response 200:**
```json
{
  "action": 38,
  "visits": { "0": 0, "1": 0, ..., "38": 12, ... },
  "time_ms": 143.7
}
```
`visits` keys are string-encoded action indices; values are visit counts from the MCTS root.

**Response 400 (terminal game):**
```json
{ "error": "Game is already terminal; no move available." }
```

**Response 503 (model not loaded):**
```json
{ "error": "MCTS model is not loaded. Set MODEL_PATH and restart the server." }
```

**Response 500 (MCTS error):**
```json
{ "error": "MCTS error: <exception message>" }
```

---

### `OPTIONS *`
Pre-flight CORS — always returns 200 with CORS headers.

---

## 8. Shared Constants and Magic Numbers

| Constant           | Value  | Where defined                          | Usage                              |
|--------------------|--------|----------------------------------------|------------------------------------|
| Board size         | `11`   | `hex_game.py: SIZE`                    | Grid dimensions, all modules       |
| Board cells        | `121`  | `11 * 11`                              | Action space size, state vector    |
| State vector size  | `122`  | `hex_state_vector.py: STATE_VECTOR_SIZE` | Obs space, ONNX input, Unity      |
| Player 1           | `1`    | Convention everywhere                  |                                    |
| Player 2           | `2`    | Convention everywhere                  |                                    |
| Empty cell         | `0`    | `hex_game.py`                          | Board encoding                     |
| P1 encoding        | `+1.0` | `hex_state_vector.py`                  | State vector cell value            |
| P2 encoding        | `-1.0` | `hex_state_vector.py`                  | State vector cell value            |
| Reward win         | `+1.0` | `hex_env.py`                           | PPO reward                         |
| Reward loss        | `-1.0` | `hex_env.py`                           | PPO reward                         |
| AI think delay     | `0.5s` | `HexBoard.cs`                          | WaitForSeconds before AI move      |
| Server host        | `localhost` | `hex_server.py: HOST`             |                                    |
| Server port        | `5000` | `hex_server.py: PORT`                  |                                    |
| Checkpoint freq    | `50,000` | `hex_train.py: CHECKPOINT_FREQ`     | Steps between checkpoint saves     |
| features_dim       | `128`  | `HexCNNExtractor`                      | Policy network output width        |
| MCTS c_puct        | `1.4`  | `hex_mcts.py: MCTS.__init__`           | PUCT exploration constant          |
| Dirichlet alpha    | `0.3`  | `hex_mcts.py`                          | Noise concentration parameter      |
| Dirichlet mix      | `0.25` | `hex_mcts.py`                          | Noise weight at root               |
| MCTS default sims  | `200`  | `hex_mcts.py: MCTS.__init__`           | Server default                     |
| Easy sims          | `8`    | `DifficultySelector.cs`                | Unity MCTS simulations             |
| Medium sims        | `100`  | `DifficultySelector.cs`                |                                    |
| Hard sims          | `400`  | `DifficultySelector.cs`                |                                    |
| Expert sims        | `1000` | `DifficultySelector.cs`                |                                    |
| GridSize (Unity)   | `11`   | `HexBoard.cs: const int GridSize`      |                                    |
| ONNX input name    | `obs_0` | `hex_export.py`, `AIManager.cs`       |                                    |
| ONNX output name   | `action_probs` | `hex_export.py`, `AIManager.cs` |                               |
| ONNX opset         | `17`   | `hex_export.py`                        |                                    |

---

## 9. Checkpoint and Model Files

### `training/models/`
| File               | Description                          |
|--------------------|--------------------------------------|
| `hex_easy.zip`     | Final easy-level SB3 PPO model       |
| `hex_easy.onnx`    | Exported ONNX actor (easy, opset 17) |

### `training/checkpoints/hard/`
| File                           | Steps     |
|--------------------------------|-----------|
| `hex_hard_50000_steps.zip`     | 50,000    |
| `hex_hard_100000_steps.zip`    | 100,000   |
| `hex_hard_150000_steps.zip`    | 150,000   |
| `hex_hard_1300000_steps.zip`   | 1,300,000 |
| `hex_hard_1350000_steps.zip`   | 1,350,000 |
| `hex_hard_1400000_steps.zip`   | 1,400,000 |

**Note:** Hard training is in progress (target: 8,000,000 steps). The checkpoints listed above are from the new residual network run (architecture changed from 3×plain-conv to 6×ResidualBlock + GAP). The easy checkpoints directory is empty (training complete, final model saved).

Checkpoint naming: `hex_{level}_{steps}_steps.zip`  
Latest checkpoint resolved by: `max(re.findall(r"\d+", filename))` in `_latest_checkpoint()`.

---

## 10. Unity Project Settings

### Unity Version
```
6000.3.11f1 (revision 3000ef702840)
```

### Key Packages (`Packages/manifest.json`)

| Package                            | Version  | Purpose                        |
|------------------------------------|----------|--------------------------------|
| `com.unity.ai.inference`           | 2.5.0    | Unity Sentis / Inference Engine (ONNX runtime) |
| `com.unity.inputsystem`            | 1.19.0   | New Input System (mouse/keyboard) |
| `com.unity.render-pipelines.universal` | 17.3.0 | URP rendering pipeline        |
| `com.unity.ugui`                   | 2.0.0    | UI Toolkit (Canvas, TextMeshPro) |
| `com.unity.timeline`               | 1.8.11   | Timeline (unused in game)      |
| `com.unity.2d.tilemap`             | 1.0.0    | Tilemap                        |
| `com.unity.test-framework`         | 1.6.0    | Unit testing                   |
| `com.unity.collab-proxy`           | 2.11.4   | Version control proxy          |
| `com.coplaydev.unity-mcp`          | git      | MCP server for Unity editor    |

### Sentis (Unity Inference Engine) Notes
- Package: `com.unity.ai.inference` **2.5.0** (formerly "Unity Sentis")
- API namespace: `Unity.InferenceEngine`
- Backend used: `BackendType.CPU` (synchronous)
- Input: `new Tensor<float>(new TensorShape(1, 122), stateVector)`
- Output: `worker.PeekOutput(outputLayerName) as Tensor<float>` → `[1, 121]`
- Configured in: `AIManager.cs`

---

## 11. Running the Project

### Start the server
```powershell
# From project root, with training venv active:
.venv-training\Scripts\Activate.ps1
python main.py --mode server

# To use a specific model for MCTS:
$env:MODEL_PATH = "training\models\hex_hard.zip"
python main.py --mode server
```

### Train a new level
```powershell
cd training
python hex_train.py --level hard
python hex_train.py --level hard --continue   # resume from saved model
```

### Export to ONNX (for Unity AIManager fallback)
```powershell
python training/hex_export.py --level easy
```

### Test the MCTS endpoint
```powershell
Invoke-WebRequest -UseBasicParsing -Uri http://localhost:5000/mcts_move `
  -Method POST -ContentType "application/json" `
  -Body '{"simulations": 100}'
```

### Unity
1. Open project in Unity 6000.3.11f1
2. Ensure the Python server is running
3. Press Play — the DifficultySelector overlay appears
4. Select difficulty → game begins; AI uses MCTS via server with fallback to local ONNX

---

## 12. Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                   Unity (C#)                        │
│                                                     │
│  HexBoard ──────── HexClient ──── UnityWebRequest   │
│     │  └─ AIManager (ONNX)            │             │
│     │       (fallback)                │             │
│  HexCell                              │             │
│  UIManager                            │             │
│  DifficultySelector                   │             │
└───────────────────────────────────────┼─────────────┘
                                        │ HTTP localhost:5000
┌───────────────────────────────────────┼─────────────┐
│              Python Server            │             │
│                                       │             │
│  hex_server.py ◄──────────────────────┘             │
│       │                                             │
│  GameManager ─── HexGame (immutable)                │
│       │                                             │
│  MCTS (hex_mcts.py)                                 │
│       │                                             │
│  PPO model (SB3, .zip) ◄─── hex_train.py            │
│  HexCNNExtractor                                    │
│  (ResNet stem + 6×ResBlock + GAP)                   │
└─────────────────────────────────────────────────────┘

Training pipeline:
  HexEnv (Gymnasium) ─→ PPO (SB3) ─→ hex_{level}.zip
                                    └─→ hex_export.py ─→ hex_{level}.onnx
```
