using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Handles difficulty selection UI: highlights the chosen button, loads the
/// matching ONNX model into AIManager, and resets the game.
///
/// Attach to the DifficultyPanel GameObject.
/// Wire aiManager, hexBoard, uiManager and the four Buttons in the Inspector
/// or via the Phase 4 scene-wiring menu item.
/// </summary>
public class DifficultySelector : MonoBehaviour
{
    // ── Inspector references ───────────────────────────────────────────────
    [Header("References")]
    public AIManager aiManager;
    public HexBoard  hexBoard;
    public UIManager uiManager;

    [Header("Difficulty Buttons")]
    public Button easyButton;
    public Button mediumButton;
    public Button hardButton;
    public Button expertButton;
    // ── Simulation counts per difficulty ─────────────────────────────────────
    private static int MctsSimulations(AIManager.Difficulty d) => d switch
    {
        AIManager.Difficulty.Easy   =>    8,
        AIManager.Difficulty.Medium =>  100,
        AIManager.Difficulty.Hard   =>  400,
        AIManager.Difficulty.Expert => 1000,
        _                           =>  100,
    };
    // ── Colour palette ─────────────────────────────────────────────────────
    private static readonly Color SelectedColor   = new Color(0.22f, 0.55f, 0.88f, 1f);
    private static readonly Color UnselectedColor = new Color(0.22f, 0.22f, 0.28f, 0.90f);

    // ── Lifecycle ──────────────────────────────────────────────────────────

    private void Start()
    {
        easyButton?.onClick.AddListener(SelectEasy);
        mediumButton?.onClick.AddListener(SelectMedium);
        hardButton?.onClick.AddListener(SelectHard);
        expertButton?.onClick.AddListener(SelectExpert);

        // Highlight default without triggering a reset yet
        HighlightSelected(AIManager.Difficulty.Medium);
    }

    // ── Public selection methods ───────────────────────────────────────────

    public void SelectEasy()   => Select(AIManager.Difficulty.Easy);
    public void SelectMedium() => Select(AIManager.Difficulty.Medium);
    public void SelectHard()   => Select(AIManager.Difficulty.Hard);
    public void SelectExpert() => Select(AIManager.Difficulty.Expert);

    // ── Private helpers ────────────────────────────────────────────────────

    private void Select(AIManager.Difficulty difficulty)
    {
        aiManager?.LoadModel(difficulty);
        hexBoard?.SetMctsSimulations(MctsSimulations(difficulty));
        HighlightSelected(difficulty);
        uiManager?.ShowDifficulty(difficulty.ToString());

        // Hide the overlay and start the game.
        uiManager?.HideDifficultySelector();
        hexBoard?.ResetGame();
    }

    private void HighlightSelected(AIManager.Difficulty difficulty)
    {
        SetButtonColor(easyButton,   difficulty == AIManager.Difficulty.Easy);
        SetButtonColor(mediumButton, difficulty == AIManager.Difficulty.Medium);
        SetButtonColor(hardButton,   difficulty == AIManager.Difficulty.Hard);
        SetButtonColor(expertButton, difficulty == AIManager.Difficulty.Expert);
    }

    private static void SetButtonColor(Button btn, bool selected)
    {
        if (btn == null) return;
        var img = btn.GetComponent<Image>();
        if (img != null)
            img.color = selected ? SelectedColor : UnselectedColor;
    }
}
