using UnityEngine;

/// <summary>
/// Attached to each Cell_[row]_[col] GameObject.
/// Manages visual state and routes mouse input to HexInputHandler.
///
/// A MeshCollider is added automatically at runtime if one is not present —
/// required for OnMouseEnter/Exit/Down to fire.
///
/// Color property name is detected from the shared material at startup so
/// both URP Unlit (_BaseColor) and legacy Unlit/Color (_Color) shaders work.
/// </summary>
[RequireComponent(typeof(MeshRenderer), typeof(MeshFilter))]
public class HexCell : MonoBehaviour
{
    // ── Public fields (set by HexBoard.FindAllCells or in the Inspector) ──
    public int row;
    public int col;
    public int action; // row * 11 + col

    // ── Cell states ───────────────────────────────────────────────────────
    public enum CellState { Empty, Player1, Player2 }

    // ── Piece / highlight colours ─────────────────────────────────────────
    private static readonly Color ColorP1Piece = new Color(0.65f, 0.08f, 0.08f, 1f); // deep red
    private static readonly Color ColorP2Piece = new Color(0.08f, 0.18f, 0.65f, 1f); // deep blue
    private static readonly Color ColorHover   = new Color(1.00f, 0.95f, 0.40f, 1f); // yellow tint
    private static readonly Color ColorLegal   = new Color(0.72f, 0.92f, 0.72f, 1f); // faint green

    // ── Private state ─────────────────────────────────────────────────────
    private MeshRenderer    meshRenderer;
    private Color           emptyColor;   // original border / interior colour
    private string          colorProp;    // "_BaseColor" (URP) or "_Color" (legacy)
    private CellState       currentState  = CellState.Empty;
    private bool            isLegal;
    private bool            isHovered;

    public bool IsLegal => isLegal;

    // ── Lifecycle ─────────────────────────────────────────────────────────

    private void Awake()
    {
        meshRenderer = GetComponent<MeshRenderer>();

        // Detect shader colour property before we ever instance the material
        var sharedMat = meshRenderer.sharedMaterial;
        colorProp  = sharedMat != null && sharedMat.HasProperty("_BaseColor")
                   ? "_BaseColor"
                   : "_Color";
        emptyColor = sharedMat != null ? sharedMat.GetColor(colorProp) : Color.white;

        // Add a MeshCollider so OnMouse* events fire (HexBoardSetup does not add one)
        if (GetComponent<MeshCollider>() == null)
        {
            var mc        = gameObject.AddComponent<MeshCollider>();
            mc.sharedMesh = GetComponent<MeshFilter>().sharedMesh;
        }
    }

    // ── Public API (called by HexBoard) ───────────────────────────────────

    /// <summary>Set the piece state and refresh the cell colour.</summary>
    public void SetState(CellState state)
    {
        currentState = state;
        isLegal      = false;
        isHovered    = false;
        RefreshColor();
    }

    /// <summary>
    /// Mark this cell as a legal move target.
    /// Only meaningful when the cell is empty — occupied cells are never legal.
    /// </summary>
    public void SetHighlight(bool legal)
    {
        isLegal   = legal;
        isHovered = isHovered && legal; // clear hover if no longer legal
        RefreshColor();
    }

    // ── Hover (called by HexBoard.Update) ───────────────────────────────────

    public void SetHover(bool hovered)
    {
        if (!isLegal) return;
        if (isHovered == hovered) return;
        isHovered = hovered;
        RefreshColor();
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private void RefreshColor()
    {
        Color c;
        if (currentState == CellState.Player1)
            c = ColorP1Piece;
        else if (currentState == CellState.Player2)
            c = ColorP2Piece;
        else if (isHovered)
            c = ColorHover;
        else if (isLegal)
            c = ColorLegal;
        else
            c = emptyColor;

        // material (not sharedMaterial) creates a per-instance copy automatically
        meshRenderer.material.SetColor(colorProp, c);
    }
}
