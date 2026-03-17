using UnityEngine;

/// <summary>
/// Receives click notifications from HexCell and forwards them to HexBoard.
/// Single responsibility: input routing only.
///
/// inputEnabled is toggled by HexBoard while waiting for Python responses,
/// preventing double-clicks during server round-trips.
///
/// Attach to the Board GameObject alongside HexBoard and HexClient.
/// Wire hexBoard in the Inspector (or let HexBoard set it at runtime).
/// </summary>
public class HexInputHandler : MonoBehaviour
{
    public HexBoard hexBoard;

    /// <summary>
    /// Set to true by HexBoard when it is safe to accept input.
    /// HexCell checks this before firing OnMouseDown.
    /// </summary>
    [HideInInspector]
    public bool inputEnabled = false;

    public void OnCellClicked(int action)
    {
        hexBoard.OnCellClicked(action);
    }
}
