# Architecture

## Core Principles

**Unity is a GUI only.** No rules logic, no AI, no game 
state lives in Unity. Unity renders what it is told to 
render and reports human input to Python. Nothing more.

**Python is the source of truth.** All rules, all legal 
move validation, all game state, all AI inference live 
in Python.

**Three core functions.** Every game module implements 
exactly three functions — get_state_vector, 
get_legal_actions, apply_action. This is the permanent 
interface between any game module and the infrastructure.

**State vector is the contract.** The state vector format 
is the API contract between Python and Unity. Defined 
before either side is built. Must never drift between 
training and inference.

## State Vector — Hex

Size: 122 floats

Positions 0-120: board cells in row-major order
  1.0  = Player 1 piece
 -1.0  = Player 2 piece
  0.0  = empty

Position 121: current player
  1.0  = Player 1 to move
 -1.0  = Player 2 to move

## Action Space — Hex

Actions are integers 0-120.
Action N = place piece on cell N in row-major order.
row = N // 11
col = N % 11

## Communication — Python to Unity

HTTP on localhost:5000
Unity never validates moves.
Python is always the source of truth.
```

**2. Add the implementation plan**

Copy your implementation plan .md file into the repo root.

**Commit both files:**
```
git add ARCHITECTURE.md wargame-ai-implementation-plan.md
git commit -m "Add architecture documentation and implementation plan"
git push