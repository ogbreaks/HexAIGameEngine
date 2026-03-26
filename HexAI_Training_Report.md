# HexAI AlphaZero Training System — Technical Report

## 1. Overview

This document describes the complete training pipeline for HexAI — an AlphaZero-style system that learns to play 11×11 Hex from scratch through self-play, with no human game data and no hand-crafted heuristics. The output is a single ONNX file containing a trained neural network that can be loaded by a game client (Unity) to play Hex at a high level.

**Key idea:** A neural network starts knowing nothing about Hex. It plays games against itself, guided by a tree search algorithm. It learns from those games. Over hundreds of iterations of "play, learn, evaluate", the network becomes progressively stronger.

The system is based on DeepMind's AlphaZero algorithm (Silver et al., 2017), adapted for Hex.

---

## 2. The Game: 11×11 Hex

Hex is a two-player strategy game played on an 11×11 rhombus-shaped grid of hexagonal cells.

- **Player 1** places pieces and tries to connect the **top edge** (row 0) to the **bottom edge** (row 10).
- **Player 2** places pieces and tries to connect the **left edge** (column 0) to the **right edge** (column 10).
- Players alternate turns; each turn consists of placing one piece on any empty cell.
- **There are no draws in Hex.** This is a proven mathematical property — one player must always win.

The board is represented internally as a flat array of 121 integers (`0` = empty, `1` = Player 1's piece, `2` = Player 2's piece). Each cell is identified by its row-major index: cell `i` corresponds to row `i // 11`, column `i % 11`.

Each hexagonal cell has up to 6 neighbours (as opposed to 4 or 8 in a square grid):
```
  upper-left:  (r-1, c)
  upper-right: (r-1, c+1)
  left:        (r,   c-1)
  right:       (r,   c+1)
  lower-left:  (r+1, c-1)
  lower-right: (r+1, c)
```

A win is detected by a breadth-first search (BFS) checking whether the last player to move has a connected path of their pieces spanning their two target edges.

---

## 3. State Representation

Before the neural network can look at a board position, the board must be converted into a fixed-size numerical vector. This is called the **state vector**.

**Format:** 122 floating-point numbers.

| Index   | Meaning                                       | Encoding      |
|---------|-----------------------------------------------|---------------|
| 0–120   | Board cells in row-major order                | +1.0 = Player 1 piece, -1.0 = Player 2 piece, 0.0 = empty |
| 121     | Current player to move                        | +1.0 = Player 1's turn, -1.0 = Player 2's turn |

This encoding is symmetric: the network always sees its own pieces as +1 and the opponent's as -1, with the current player indicator telling it whose turn it is. This allows a single network to play for both sides.

---

## 4. Neural Network Architecture

The network is called a **Policy-Value Network**. It takes a board state as input and produces two outputs simultaneously:

1. **Policy** (121 numbers): A probability distribution over all 121 cells, indicating how promising each move looks. Higher values mean the network considers that move stronger.
2. **Value** (1 number): A score between -1.0 and +1.0 estimating the current player's chance of winning from this position. +1.0 = certain win, -1.0 = certain loss, 0.0 = even.

### 4.1 Trunk (Shared Feature Extractor)

The input vector is reshaped into an 11×11 single-channel image (like a greyscale photograph of the board). This image passes through a **residual network (ResNet)** trunk:

1. **Stem layer:** A 3×3 convolution that expands the single-channel input to 128 feature channels, followed by batch normalisation and a ReLU activation. Think of this as the network learning 128 different "ways to look at" the raw board.

2. **Residual tower:** 10 sequential residual blocks. Each block contains:
   - A 3×3 convolution → batch normalisation → ReLU
   - Another 3×3 convolution → batch normalisation
   - A **skip connection** that adds the block's input directly to its output, then applies ReLU

   The skip connection is the key innovation of ResNets — it allows the network to be deep (many layers) without suffering from the "vanishing gradient" problem where training signals disappear in deep networks. Each block can learn to refine the features from the previous block, or simply pass them through unchanged.

The trunk's output is a tensor of shape `[batch, 128, 11, 11]` — 128 learned feature maps, each preserving the spatial 11×11 structure of the board.

**Default configuration:** 10 residual blocks × 128 channels ≈ 10 million trainable parameters.

### 4.2 Policy Head

Converts the trunk's spatial features into move probabilities:

1. 1×1 convolution reducing 128 channels → 2 channels (compression)
2. Batch normalisation → ReLU
3. Flatten to a 242-element vector (2 × 11 × 11)
4. Fully connected layer → 121 outputs (one per board cell)

The output is **raw logits** (unnormalised scores). They are later converted to probabilities via softmax: $P(a) = \frac{e^{z_a}}{\sum_{j} e^{z_j}}$ where $z_a$ is the logit for action $a$.

### 4.3 Value Head

Converts the trunk's spatial features into a single win-probability estimate:

1. 1×1 convolution reducing 128 channels → 1 channel
2. Batch normalisation → ReLU
3. Flatten to 121 elements
4. Fully connected layer → 64 hidden units → ReLU
5. Fully connected layer → 1 output → **tanh** activation

The tanh function squashes the output to the range [-1, +1], which matches the value convention: +1 = win, -1 = loss.

### 4.4 Why Two Heads?

Sharing a trunk between policy and value heads is not just an efficiency trick — it forces the network to learn features that are useful for *both* move selection and position evaluation. This shared representation is richer than two separate networks would learn independently.

---

## 5. Monte Carlo Tree Search (MCTS)

MCTS is the "thinking" algorithm. Instead of the neural network choosing moves directly (which would be weak), MCTS uses the network's estimates to conduct a structured lookahead search, considering thousands of possible future positions before deciding on a move.

### 5.1 The Search Tree

MCTS builds a tree of game positions in memory. Each node stores:
- The game state (board position)
- **Visit count** $N$: how many times this node has been explored
- **Value sum** $W$: accumulated value from all simulations passing through this node
- **Prior probability** $P$: the network's initial assessment of how good the move leading to this node is
- **Mean value** $Q = W / N$: average outcome from this node's perspective

### 5.2 One Simulation

Each simulation has four phases:

#### Phase 1: Selection
Starting from the root (current game position), descend through the tree by repeatedly picking the child with the highest **PUCT score**:

$$\text{UCB}(s, a) = Q(s,a) + c_{\text{puct}} \cdot P(s,a) \cdot \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

Where:
- $Q(s,a)$ = mean value of taking action $a$ from state $s$ (exploitation)
- $P(s,a)$ = the network's prior probability for this move (guides initial exploration)
- $N(s)$ = total visits to the parent node
- $N(s,a)$ = visits to this specific child
- $c_{\text{puct}}$ = exploration constant (1.4), controlling the trade-off between exploitation and exploration

**Intuition:** Moves with high $Q$ (historically strong outcomes) score well. Moves with high $P$ (the network thinks they're good) also score well. Moves with low visit count $N(s,a)$ get a bonus from the $\frac{1}{1+N}$ term, ensuring under-explored moves get sampled. Over many simulations, the search converges on the strongest move while still exploring alternatives.

#### Phase 2: Expansion
When selection reaches a node that hasn't been expanded yet (a "leaf"), the neural network evaluates that position:
- The **policy output** creates child nodes for all legal moves, each with a prior probability.
- The **value output** provides an estimate of who's winning from this position.

The priors are renormalised to sum to 1.0 over legal moves only (illegal moves are discarded).

#### Phase 3: Backpropagation
The value estimate is propagated back up the tree from the leaf to the root. At each level, the value is **negated** (because Hex is a zero-sum game — what's good for one player is bad for the other). Each node on the path has its visit count incremented and its value sum updated.

#### Phase 4: Get Policy
After all simulations complete (typically 200–800), the root node's children have accumulated visit counts. The **visit count distribution** becomes the MCTS policy — moves that were visited more often are considered stronger:

$$\pi(a) = \frac{N(\text{root}, a)^{1/\tau}}{\sum_b N(\text{root}, b)^{1/\tau}}$$

Where $\tau$ is the **temperature** parameter:
- $\tau = 1.0$: proportional to visit counts (used early in self-play for exploration)
- $\tau \to 0$: picks the most-visited move (used for evaluation/competition)

### 5.3 Virtual Loss (Batched MCTS)

To utilise the GPU efficiently, multiple MCTS simulations are run in parallel within a single search. The problem is that naïve parallelism would send all threads down the same path (the current best). **Virtual loss** solves this:

Before descending a path, a pessimistic penalty (-1.0) is applied to all nodes on that path. This temporarily makes the path look worse, encouraging the next parallel simulation to explore a *different* path. After the neural network evaluates the leaf, the penalty is corrected with the real value.

This allows batching $K$ board evaluations into a single GPU forward pass (e.g., $K = 8$), dramatically improving throughput.

### 5.4 Dirichlet Noise

During self-play (training data generation), random noise is added to the root node's prior probabilities:

$$P'(a) = (1 - \epsilon) \cdot P(a) + \epsilon \cdot \eta_a, \quad \eta \sim \text{Dir}(\alpha)$$

Where $\alpha = 0.3$ and $\epsilon = 0.25$. This ensures the search explores moves the network might not initially favour, which is critical for discovering new strategies during training.

---

## 6. The Training Loop (AlphaZero)

Training is an iterative cycle. Each **iteration** consists of three phases:

### 6.1 Phase 1: Self-Play Data Generation

The current best network plays games against itself to produce training data.

1. Multiple CPU worker processes are spawned (e.g., 8 workers × 20 games each = 160 games per iteration).
2. Each worker gets a copy of the best network.
3. For each game:
   - Starting from an empty board, MCTS is used to select every move.
   - Early moves use temperature $\tau = 1.0$ (exploration); after 30 moves, $\tau = 0.01$ (near-greedy).
   - Dirichlet noise is added at the root of every MCTS search.
   - The game plays to completion (someone wins — no draws in Hex).
4. Each position in the game is recorded as a training example:
   - **Input:** the state vector (122 floats)
   - **Policy target:** the MCTS visit-count distribution (121 floats)
   - **Value target:** +1.0 if the player to move at this position eventually won, -1.0 if they lost

All training examples are added to a **replay buffer** — a large FIFO queue (up to 500,000 entries). This means the network trains on a mix of recent and slightly older data, which improves stability.

**Symmetry augmentation (4× data):** Before entering the replay buffer, each training example is augmented with three additional copies produced by applying the symmetry group of the Hex board. 11×11 Hex has a symmetry group of order 4:

1. **Identity** — the original example (already present)
2. **180° rotation** — $(r,c) \to (10-r, 10-c)$, no colour swap needed (both players' goal edges map back to themselves)
3. **Diagonal transpose** — $(r,c) \to (c,r)$, with colour swap (Player 1's top↔bottom path becomes a left↔right path, which is Player 2's goal)
4. **Anti-diagonal transpose** — $(r,c) \to (10-c, 10-r)$, with colour swap

All four transforms preserve the hex adjacency structure — every neighbour pair maps to another valid neighbour pair. Each transform is an involution (applying it twice returns the original), and the group composes correctly: diagonal ∘ anti-diagonal = 180° rotation.

The state vector, policy vector, and current-player indicator are all transformed consistently. The value target is unchanged (the position's strategic evaluation is preserved under symmetry). This quadruples effective training data with zero additional self-play compute — equivalent to running self-play 4× longer.

This feature is controlled by the `augment_symmetry` configuration flag and can be disabled for fast test runs.

**GPU Inference Server (optional):** When running on a GPU, a dedicated inference server process handles all neural network evaluations. Workers send board states to the server via shared memory queues; the server batches them into efficient GPU forward passes and returns results. This can improve throughput by 5–10× compared to CPU-only workers.

### 6.2 Phase 2: Network Training

Once enough data has accumulated in the replay buffer (minimum 10,000 entries), the network is trained:

1. **Sampling:** Random mini-batches (512 examples) are drawn from the replay buffer.
2. **Forward pass:** Each batch of state vectors is fed through the network to get policy logits and value predictions.
3. **Loss computation:** Two losses are computed:
   - **Policy loss:** Cross-entropy between the network's policy output and the MCTS visit-count targets. Illegal moves are masked out (their logits are set to $-10^9$ before softmax).

     $$L_\pi = -\sum_a \pi_{\text{MCTS}}(a) \log P_\theta(a)$$

   - **Value loss:** Mean squared error between the network's value output and the actual game outcome.

     $$L_v = (v_\theta - z)^2$$

   - **Total loss:** $L = L_\pi + L_v$
4. **Backpropagation:** Gradients are computed and the network's weights are updated using the Adam optimiser.
5. This is repeated for 100 gradient steps per iteration.

**Learning rate schedule:** Cosine annealing, starting at 0.001 and decaying to near-zero over the full training run. This means the network makes large updates early (when there's a lot to learn) and fine-grained adjustments later.

### 6.3 Phase 3: Arena Evaluation

Every 10 iterations, the updated network (**challenger**) plays a tournament against the current best network (**champion**):

1. 40 games are played: 20 with the challenger as Player 1, 20 as Player 2 (to eliminate first-player advantage bias).
2. Both players use MCTS with 200 simulations per move (no noise, near-greedy temperature — pure evaluation).
3. If the challenger wins more than 55% of games, it is **promoted** to become the new champion.
4. If not, the champion is retained and the challenger is discarded.

This gating mechanism ensures that the "best network" used for self-play data is monotonically non-decreasing in strength. A bad training episode cannot corrupt the self-play data distribution.

**ELO rating:** After each arena evaluation, an ELO rating is computed for the surviving network using the standard formula:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

$$R'_A = R_A + K \cdot (S_A - E_A)$$

Where $K = 32$ is the sensitivity factor and $S_A$ is the observed score (win rate). The ELO starts at 1000 for the untrained network and increases as the network improves. This provides an absolute strength metric across the entire training run — unlike loss curves, a rising ELO reliably indicates the agent is getting stronger.

### 6.4 Phase 4: Checkpointing

Every 50 iterations (and at the end of training), a full checkpoint is saved containing:
- Network weights
- Optimiser state (momentum accumulators)
- Learning rate scheduler state
- Current iteration number
- Configuration

This allows training to be **resumed** from any checkpoint if interrupted (e.g., due to cloud preemption or hardware failure).

When a network is promoted to champion, an **ONNX export** is also saved — this is the deployable model file.

---

## 7. Complete Iteration Walkthrough

Here is exactly what happens during one iteration (e.g., iteration 30 of 100):

```
1. SELF-PLAY PHASE
   └─ 8 CPU workers each play 20 games (160 total)
   └─ Each game: ~60–80 moves, each requiring 200 MCTS simulations
   └─ Produces ~10,000 labelled (state, policy, value) training examples
   └─ Symmetry augmentation: 10,000 × 4 = ~40,000 entries
   └─ Added to replay buffer (now contains ~150,000 examples)

2. TRAINING PHASE
   └─ 100 gradient steps
   └─ Each step: sample 512 examples → forward pass → compute loss → backprop → update weights
   └─ Losses decrease (policy ≈ 1.5 → 1.2, value ≈ 0.8 → 0.6)

3. ARENA PHASE (every 10th iteration)
   └─ Challenger (iter 30 weights) vs Champion (last promoted weights)
   └─ 40 games, split first-mover fairly
   └─ Result: 26W 14L → 65% win rate > 55% threshold → PROMOTED
   └─ ELO: 1000 → 1004.8
   └─ ONNX exported: hex_az_30.onnx

4. CHECKPOINT
   └─ Saved: hex_az_30.pth (full training state)
   └─ Saved: hex_az_best.pth (promoted weights only)

5. METRICS UPDATE
   └─ training/metrics.json updated (dashboard reads this)
   └─ training/elo_state.json updated
```

---

## 8. Configuration

All hyperparameters are controlled via YAML configuration files. The default configuration (`config/hex11_default.yaml`):

| Parameter               | Default     | Purpose                                                |
|-------------------------|-------------|--------------------------------------------------------|
| `trunk`                 | resnet      | Network architecture type                              |
| `num_res_blocks`        | 10          | Depth of the residual tower                            |
| `num_channels`          | 128         | Width of all convolutional layers                      |
| `num_simulations`       | 800         | MCTS simulations per move during self-play             |
| `c_puct`                | 1.4         | Exploration constant in PUCT formula                   |
| `dirichlet_alpha`       | 0.3         | Noise concentration for root exploration               |
| `dirichlet_weight`      | 0.25        | How much noise vs. network prior at root               |
| `temperature_threshold` | 30          | Move number after which temperature drops to ~0        |
| `num_workers`           | 4           | Parallel self-play processes                           |
| `games_per_worker`      | 25          | Games each worker plays per iteration                  |
| `augment_symmetry`      | true        | 4× data augmentation via Hex board symmetries          |
| `num_iterations`        | 1000        | Total training iterations                              |
| `train_steps_per_iter`  | 100         | Gradient updates per iteration                         |
| `batch_size`            | 512         | Training mini-batch size                               |
| `lr_init`               | 0.001       | Initial learning rate (Adam)                           |
| `min_buffer_size`       | 10,000      | Minimum replay buffer entries before training starts   |
| `buffer_size`           | 500,000     | Maximum replay buffer capacity (FIFO eviction)         |
| `arena_freq`            | 10          | Evaluate every N iterations                            |
| `arena_games`           | 40          | Games per arena evaluation                             |
| `arena_simulations`     | 200         | MCTS simulations during arena (lower = faster)         |
| `promotion_threshold`   | 0.55        | Win rate required to promote challenger                |
| `elo_k_factor`          | 32          | ELO rating sensitivity per arena evaluation            |
| `elo_initial`           | 1000        | Starting ELO for untrained network                     |
| `checkpoint_freq`       | 50          | Save full checkpoint every N iterations                |

A separate cloud-optimised configuration (`config/hex11_t4.yaml`) is tuned for a GCE T4 GPU instance with the inference server enabled, targeting a strong agent in ~2 hours of compute.

---

## 9. Infrastructure

### 9.1 Execution Environment

Training runs inside a Docker container (`pixelpunk77/hexai-az:latest`) on a Google Cloud Compute Engine (GCE) VM with an NVIDIA T4 GPU. The startup script (`startup-gpu.sh`):

1. Mounts a persistent disk for model storage (survives VM shutdown)
2. Installs Docker and the NVIDIA container toolkit
3. Pulls and runs the training container with GPU access
4. On exit (success or failure), automatically stops the VM to avoid runaway costs

### 9.2 Monitoring Dashboard

A FastAPI web server (`metrics_server.py`) runs alongside training on port 8080, serving:

- `/dashboard` — A real-time HTML dashboard showing:
  - Training progress (iteration count, ETA, percent complete)
  - Current training phase (self-play / training / arena / checkpoint)
  - Loss curves (policy loss, value loss)
  - Network ELO rating with sparkline trend
  - Cost tracking (hourly rate × elapsed time)
  - Hardware utilisation (per-core CPU bars, GPU utilisation/VRAM/temperature)
  - Event log (timestamped training events)
- `/metrics` — Raw JSON metrics (consumed by the dashboard)
- `/hardware` — Live CPU/GPU utilisation data

Metrics are written atomically (write to temp file, then `os.replace`) to prevent partial reads.

### 9.3 Cost Management

The system tracks accumulated training hours across sessions in `training/cost_state.json`. Combined with the configured hourly rate (e.g., $0.50/hr for a T4 spot instance), it provides real-time and projected total cost estimates on the dashboard.

---

## 10. Output Artefacts

At the end of training, the following files are produced:

| File                          | Contents                                                        | Used By         |
|-------------------------------|-----------------------------------------------------------------|-----------------|
| `hex_az_best.pth`            | PyTorch weights of the strongest network (best champion)        | Resume training |
| `hex_az_best.onnx`           | ONNX export of the policy head only                             | Unity game      |
| `hex_az_{iter}.onnx`         | ONNX snapshots from each promotion                              | Comparison      |
| `hex_az_{iter}.pth`          | Full checkpoints (weights + optimiser + scheduler)              | Resume training |
| `hex_az_final.pth`           | Final checkpoint at end of training                             | Resume training |
| `elo_state.json`             | ELO rating history across all arena evaluations                 | Dashboard       |
| `metrics.json`               | Latest training metrics snapshot                                | Dashboard       |
| `events.json`                | Rolling event log (last 30 entries)                             | Dashboard       |

### 10.1 ONNX Export Details

The ONNX file contains **only the policy head** (the value head is discarded). Unity's Sentis inference engine loads this file.

- **Input:** `obs_0` — shape `[1, 122]`, float32 (the state vector)
- **Output:** `action_probs` — shape `[1, 121]`, float32 (raw logits; softmax applied by Unity)
- **ONNX opset:** 17

The value head is not needed at inference time because the game client does not run MCTS — it simply takes the network's top action. (A more sophisticated client could run MCTS using the value head, but the current design prioritises fast response times.)

---

## 11. Why This Works

The elegance of AlphaZero is in the feedback loop:

1. **Better network → better MCTS policy → better training data → better network**

The network's policy guides the search (which moves to explore), and the search's visit counts correct the network's mistakes (moves the search found to be strong, even if the network initially undervalued them). The value head provides a learned evaluation function that replaces the need for game-specific heuristics.

Because Hex has no draws, every self-play game produces a clear winner, providing unambiguous training signal. The replay buffer ensures the network sees diverse positions, and the arena gating prevents quality regression.

Over hundreds of iterations, the network learns strategic concepts — bridge patterns, ladder attacks, connection templates — entirely from the statistics of self-play, without any human knowledge of Hex strategy being encoded into the system.

---

## Appendix A: Glossary

| Term               | Definition                                                                                  |
|--------------------|---------------------------------------------------------------------------------------------|
| **Backpropagation**| The algorithm for computing gradients of the loss function with respect to network weights   |
| **Batch norm**     | A layer that normalises its inputs across the batch, stabilising training                   |
| **Cross-entropy**  | A loss function measuring how different two probability distributions are                   |
| **FIFO**           | First In, First Out — oldest entries are removed when the buffer is full                    |
| **Gradient**       | The direction and magnitude of change that would reduce the loss function                   |
| **Logit**          | A raw, unnormalised score output by a neural network layer                                  |
| **ONNX**           | Open Neural Network Exchange — a portable format for trained neural network models          |
| **ReLU**           | Rectified Linear Unit: $f(x) = \max(0, x)$. A simple activation function                   |
| **Softmax**        | Converts logits to probabilities: normalises so all values are positive and sum to 1        |
| **Tanh**           | Hyperbolic tangent: squashes values to [-1, +1]                                             |
| **Tensor**         | A multi-dimensional array of numbers (generalisation of vectors and matrices)               |
