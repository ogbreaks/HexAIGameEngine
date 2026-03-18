using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// Central controller for the Hex board.
///
/// Responsibilities:
///   - Locate all 121 HexCell GameObjects at startup (adds HexCell component if missing)
///   - Fetch initial state from Python on Start
///   - Route move clicks to Python, render responses
///   - Detect and announce game-over
///   - Coordinate HexClient, HexCell, and UIManager
///
/// Attach to the Board GameObject alongside HexClient and HexInputHandler.
/// Wire hexClient and uiManager in the Inspector.
/// </summary>
public class HexBoard : MonoBehaviour
{
    // ── Inspector references ──────────────────────────────────────────────
    public HexClient  hexClient;
    public UIManager  uiManager;

    [Header("AI")]
    public AIManager aiManager;
    /// <summary>When true the human plays as Player 1 (Red); AI is Player 2 (Blue).</summary>
    public bool humanIsPlayer1 = true;

    /// <summary>Number of MCTS simulations per AI move. Set by DifficultySelector.</summary>
    private int _mctsSimulations = 100;

    // ── Constants ─────────────────────────────────────────────────────────
    private const int GridSize = 11;

    // ── Runtime state ─────────────────────────────────────────────────────
    private HexCell[,]      cells        = new HexCell[GridSize, GridSize];
    private HexInputHandler inputHandler;
    private GameStateData   currentState;
    private bool            gameOver     = false;
    private HexCell         hoveredCell;


    // ── Lifecycle ─────────────────────────────────────────────────────────

    private void Start()
    {
        inputHandler = GetComponent<HexInputHandler>();

        FindAllCells();
        SpawnHexEdgeHighlights();

        uiManager?.ShowConnecting();

        StartCoroutine(hexClient.GetState(
            state =>
            {
                ApplyState(state);
                // When an AIManager is present, wait for DifficultySelector
                // to call ResetGame() before enabling input.
                if (aiManager == null)
                {
                    uiManager?.ShowTurn(state.current_player);
                    inputHandler.inputEnabled = true;
                }
            },
            err =>
            {
                uiManager?.ShowError(
                    "Cannot connect to server. Please start: python main.py --mode server");
            }));
    }

    private void Update()
    {
        if (!inputHandler.inputEnabled || gameOver) return;

        var cam = Camera.main;
        if (cam == null) return;

        var mouse = Mouse.current;
        if (mouse == null) return;

        var screenPos = mouse.position.ReadValue();
        var ray = cam.ScreenPointToRay(screenPos);

        // Hover highlight
        HexCell hitCell = null;
        if (Physics.Raycast(ray, out RaycastHit hit))
            hitCell = hit.collider.GetComponent<HexCell>();

        if (hitCell != hoveredCell)
        {
            hoveredCell?.SetHover(false);
            hoveredCell = hitCell;
            hoveredCell?.SetHover(true);
        }

        // Click
        if (mouse.leftButton.wasPressedThisFrame && hitCell != null && hitCell.IsLegal)
            OnCellClicked(hitCell.action);
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// <summary>
    /// Called by HexInputHandler when a human clicks a legal cell.
    /// Disables input, sends the move to Python, renders the response.
    /// </summary>
    public void OnCellClicked(int action)
    {
        if (gameOver) return;

        inputHandler.inputEnabled = false;

        StartCoroutine(hexClient.PostMove(action,
            state =>
            {
                ApplyState(state);

                if (state.is_terminal)
                    HandleGameOver(state);
                else
                    TakeTurn(state);
            },
            err =>
            {
                uiManager?.ShowError(err);
                inputHandler.inputEnabled = true;
            }));
    }

    /// <summary>Reset the game via Python and re-render the empty board.</summary>
    public void ResetGame()
    {
        gameOver                  = false;
        inputHandler.inputEnabled = false;

        // Immediately clear all cell colours so the old game vanishes at once.
        foreach (var cell in cells)
        {
            if (cell == null) continue;
            cell.SetState(HexCell.CellState.Empty);
            cell.SetHighlight(false);
        }

        uiManager?.ShowConnecting();

        StartCoroutine(hexClient.PostReset(
            state =>
            {
                ApplyState(state);
                TakeTurn(state);
            },
            err => uiManager?.ShowError(err)));
    }

    // ── TakeTurn / AI turn ────────────────────────────────────────────────

    /// <summary>
    /// Determines whose turn it is and either enables human input or triggers
    /// the AI coroutine.  Call this after every successful move and after reset.
    /// </summary>
    public void TakeTurn(GameStateData state)
    {
        uiManager?.ShowTurn(state.current_player);

        bool humanTurn = aiManager == null ||
            (humanIsPlayer1 ? state.current_player == 1 : state.current_player == 2);

        if (humanTurn)
        {
            inputHandler.inputEnabled = true;
        }
        else
        {
            inputHandler.inputEnabled = false;
            StartCoroutine(TakeAITurnWithDelay(state));
        }
    }

    private IEnumerator TakeAITurnWithDelay(GameStateData state)
    {
        // Brief pause so the human can see the board update before the AI moves.
        yield return new WaitForSeconds(0.5f);

        if (gameOver)
        {
            inputHandler.inputEnabled = true;
            yield break;
        }

        var legal = new List<int>(state.legal_actions ?? new int[0]);
        if (legal.Count == 0)
        {
            inputHandler.inputEnabled = true;
            yield break;
        }

        // ── Try MCTS via server ──────────────────────────────────────────────────
        int action = -1;
        Debug.Log($"[MCTS] Attempting MCTS move with {_mctsSimulations} simulations");
        yield return StartCoroutine(hexClient.PostMctsMove(
            _mctsSimulations,
            a =>
            {
                action = a;
                Debug.Log($"[MCTS] MCTS returned action {a}");
            },
            err =>
            {
                Debug.LogWarning($"[MCTS] MCTS failed, falling back: {err}");
            }
        ));

        // ── Fallback to local AIManager when the server is unavailable ────────
        if (action < 0)
        {
            if (aiManager != null)
                action = aiManager.GetBestAction(state.state_vector, legal);
            else
                action = legal[UnityEngine.Random.Range(0, legal.Count)];
        }

        StartCoroutine(hexClient.PostMove(action,
            nextState =>
            {
                ApplyState(nextState);
                if (nextState.is_terminal)
                    HandleGameOver(nextState);
                else
                    TakeTurn(nextState);
            },
            err =>
            {
                uiManager?.ShowError(err);
                inputHandler.inputEnabled = true;
            }));
    }

    // ── Training support ──────────────────────────────────────────────────

    /// <summary>Returns the most recent game state (read-only).</summary>
    public GameStateData GetCurrentState() => currentState;

    /// <summary>
    /// Set the number of MCTS simulations used for AI turns.
    /// Called by DifficultySelector when the player picks a difficulty.
    /// </summary>
    public void SetMctsSimulations(int simulations) => _mctsSimulations = simulations;

    // ── Private helpers ───────────────────────────────────────────────────

    private void ApplyState(GameStateData state)
    {
        currentState = state;
        RenderState(state);
    }

    /// <summary>
    /// Read the 122-float state vector and update every HexCell's visual state.
    ///   +1.0 → Player 1 piece
    ///   -1.0 → Player 2 piece
    ///    0.0 → empty (legal moves highlighted in green)
    /// </summary>
    private void RenderState(GameStateData state)
    {
        var legalSet = new HashSet<int>(state.legal_actions ?? new int[0]);

        for (int r = 0; r < GridSize; r++)
        for (int c = 0; c < GridSize; c++)
        {
            var cell = cells[r, c];
            if (cell == null) continue;

            float value = state.state_vector[r * GridSize + c];

            if (value > 0.5f)
            {
                cell.SetState(HexCell.CellState.Player1);
            }
            else if (value < -0.5f)
            {
                cell.SetState(HexCell.CellState.Player2);
            }
            else
            {
                cell.SetState(HexCell.CellState.Empty);
                cell.SetHighlight(legalSet.Contains(cell.action));
            }
        }
    }

    private void HandleGameOver(GameStateData state)
    {
        gameOver                  = true;
        inputHandler.inputEnabled = false;
        uiManager?.ShowWinner(state.winner);
        uiManager?.ShowDifficultySelector(); // offer difficulty re-selection for replay

        // Clear all legal-move highlights
        foreach (var cell in cells)
            cell?.SetHighlight(false);
    }

    /// <summary>
    /// Locate every Cell_[r]_[c] under Board/Cells.
    /// Adds a HexCell component and sets its row/col/action if not already present.
    /// </summary>
    private void FindAllCells()
    {
        var cellsRoot = transform.Find("Cells");
        if (cellsRoot == null)
        {
            Debug.LogError(
                "[HexBoard] 'Cells' child not found. Run Hex AI ▶ Build Board first.");
            return;
        }

        for (int r = 0; r < GridSize; r++)
        for (int c = 0; c < GridSize; c++)
        {
            var t = cellsRoot.Find($"Cell_{r}_{c}");
            if (t == null)
            {
                Debug.LogWarning($"[HexBoard] Missing Cell_{r}_{c}");
                continue;
            }

            var hexCell = t.GetComponent<HexCell>();
            if (hexCell == null)
            {
                hexCell        = t.gameObject.AddComponent<HexCell>();
                hexCell.row    = r;
                hexCell.col    = c;
                hexCell.action = r * GridSize + c;
            }

            cells[r, c] = hexCell;
        }
    }

    /// <summary>
    /// Draws thin coloured highlights along the outer edges of border hexagons so
    /// players always know which direction they must connect, even when the cells
    /// themselves are coloured green (legal move) or a piece colour.
    ///
    /// Colour scheme matches HexBoardSetup:
    ///   Red  (P1) — top row top-edges + bottom row bottom-edges
    ///   Blue (P2) — left col left-edge + right col right-edge
    ///
    /// Corner cells receive three highlighted edges (two from one player, one from
    /// the other).
    /// </summary>
    private void SpawnHexEdgeHighlights()
    {
        if (cells[0, 0] == null || cells[0, 1] == null) return;

        // Derive the rendered hex circumradius from actual cell world positions.
        // ColStep = √3 * HexSize, HexVisual ≈ HexSize * 0.94
        float colStep = Vector3.Distance(
            cells[0, 1].transform.position,
            cells[0, 0].transform.position);
        float hexR     = colStep / Mathf.Sqrt(3f) * 0.94f;
        float thickness = hexR * 0.28f;
        float zOff      = -0.05f;   // render in front of hex face

        // Match HexBoardSetup palette
        var red  = new Color(0.78f, 0.42f, 0.38f, 1f);
        var blue = new Color(0.36f, 0.52f, 0.74f, 1f);

        // Pointy-top hex vertices relative to centre, starting at top (90°), CCW.
        // v[0]=top  v[1]=upper-right  v[2]=lower-right
        // v[3]=bottom  v[4]=lower-left  v[5]=upper-left
        var v = new Vector2[6];
        for (int i = 0; i < 6; i++)
        {
            float a = Mathf.Deg2Rad * (60f * i + 90f);
            v[i] = new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * hexR;
        }

        for (int r = 0; r < GridSize; r++)
        for (int c = 0; c < GridSize; c++)
        {
            bool top    = r == 0;
            bool bottom = r == GridSize - 1;
            bool left   = c == 0;
            bool right  = c == GridSize - 1;

            if (!top && !bottom && !left && !right) continue;

            var cell = cells[r, c];
            if (cell == null) continue;

            Vector3 center = cell.transform.position;

            // P1 = red: top row exposes upper edges, bottom row exposes lower edges
            if (top)
            {
                SpawnEdgeQuad(center, v[5], v[0], red, thickness, zOff);
                SpawnEdgeQuad(center, v[0], v[1], red, thickness, zOff);
            }
            if (bottom)
            {
                SpawnEdgeQuad(center, v[2], v[3], red, thickness, zOff);
                SpawnEdgeQuad(center, v[3], v[4], red, thickness, zOff);
            }

            // P2 = blue: left col outer-left edge (v[2]→v[1]), right col outer-right edge (v[5]→v[4])
            // Each side also has a diagonal edge; skip it on corners that already drew it as red.
            if (left)
            {
                SpawnEdgeQuad(center, v[2], v[1], blue, thickness, zOff); // vertical left
                if (!bottom) // bottom-left corner: v[3]→v[2] already drawn red
                    SpawnEdgeQuad(center, v[3], v[2], blue, thickness, zOff); // lower-left diagonal
            }
            if (right)
            {
                SpawnEdgeQuad(center, v[5], v[4], blue, thickness, zOff); // vertical right
                if (!top) // top-right corner: v[0]→v[5] already drawn red
                    SpawnEdgeQuad(center, v[0], v[5], blue, thickness, zOff); // upper-right diagonal
            }
        }
    }

    private void SpawnEdgeQuad(
        Vector3 cellCenter, Vector2 va, Vector2 vb,
        Color color, float thickness, float zOff)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
        go.name = "EdgeHighlight";
        go.transform.SetParent(transform, worldPositionStays: true);
        Destroy(go.GetComponent<Collider>());

        Vector2 mid   = (va + vb) * 0.5f;
        Vector2 dir   = (vb - va).normalized;
        float   len   = (vb - va).magnitude * 1.18f;  // overshoot endpoints so adjacent lines blend
        float   angle = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;

        go.transform.position   = cellCenter + new Vector3(mid.x, mid.y, zOff);
        go.transform.rotation   = Quaternion.AngleAxis(angle, Vector3.forward);
        go.transform.localScale = new Vector3(len, thickness, 1f);

        var mr  = go.GetComponent<MeshRenderer>();
        var mat = new Material(cells[0, 0].GetComponent<MeshRenderer>().sharedMaterial);
        mat.SetColor(mat.HasProperty("_BaseColor") ? "_BaseColor" : "_Color", color);
        mr.material = mat;
    }
}
