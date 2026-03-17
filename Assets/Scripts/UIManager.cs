using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.UI;
using UnityEngine.UI;

/// <summary>
/// Handles all UI display: turn indicator, status messages, win announcement,
/// and the reset button.
///
/// Attach to the Canvas (or a dedicated UI GameObject).
/// Wire turnText, statusText, and resetButton in the Inspector.
/// </summary>
public class UIManager : MonoBehaviour
{
    // ── Inspector references ──────────────────────────────────────────────
    public TextMeshProUGUI turnText;
    public TextMeshProUGUI statusText;
    public Button          resetButton;

    [Header("Difficulty")]
    public GameObject      difficultyPanel;
    public TextMeshProUGUI difficultyText;

    // ── Lifecycle ─────────────────────────────────────────────────────────

    private void Start()
    {
        EnsureInputSystemUIModule();

        if (resetButton != null)
            resetButton.onClick.AddListener(OnResetClicked);

        // Show the difficulty chooser overlay at startup so the player
        // selects a difficulty before the first human input is allowed.
        ShowDifficultySelector();
    }

    // Replace legacy StandaloneInputModule with InputSystemUIInputModule so
    // UI buttons respond to clicks when the project uses the Input System package.
    // Also creates the EventSystem GameObject if none exists in the scene.
    private static void EnsureInputSystemUIModule()
    {
        var es = EventSystem.current;
        if (es == null)
        {
            // No EventSystem in scene — create one with the correct module.
            var go = new GameObject("EventSystem");
            go.AddComponent<EventSystem>();
            go.AddComponent<InputSystemUIInputModule>();
            return;
        }

        // EventSystem exists — swap out the legacy module if present.
        var legacy = es.GetComponent<StandaloneInputModule>();
        if (legacy != null)
        {
            Object.Destroy(legacy);
            if (es.GetComponent<InputSystemUIInputModule>() == null)
                es.gameObject.AddComponent<InputSystemUIInputModule>();
        }
    }

    // ── Public API (called by HexBoard) ───────────────────────────────────

    public void ShowTurn(int currentPlayer)
    {
        if (turnText != null)
            turnText.text = currentPlayer == 1 ? "Player 1's Turn" : "Player 2's Turn";
        if (statusText != null)
            statusText.text = string.Empty;
    }

    public void ShowWinner(int winner)
    {
        if (turnText != null)
            turnText.text = winner == 1 ? "Player 1 Wins!" : "Player 2 Wins!";
        if (statusText != null)
            statusText.text = string.Empty;
    }

    public void ShowConnecting()
    {
        if (statusText != null)
            statusText.text = "Connecting to server...";
    }

    public void ShowError(string message)
    {
        if (statusText != null)
            statusText.text = message;
    }

    public void ShowDifficulty(string difficulty)
    {
        if (difficultyText != null)
            difficultyText.text = $"Difficulty: {difficulty}";
    }

    public void ShowDifficultySelector()
    {
        if (difficultyPanel != null)
            difficultyPanel.SetActive(true);
    }

    public void HideDifficultySelector()
    {
        if (difficultyPanel != null)
            difficultyPanel.SetActive(false);
    }

    // ── Button callback ───────────────────────────────────────────────────

    public void OnResetClicked()
    {
        var board = FindFirstObjectByType<HexBoard>();
        board?.ResetGame();
    }
}
