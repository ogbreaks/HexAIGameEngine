using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Handles all HTTP communication with the Python Hex server on localhost:5000.
/// Every public method is a coroutine — never blocks the main thread.
/// </summary>
public class HexClient : MonoBehaviour
{
    private const string BASE_URL = "http://localhost:5000";

    // ── GET /state ────────────────────────────────────────────────────────
    /// <summary>Fetch the current game state from Python.</summary>
    public IEnumerator GetState(
        System.Action<GameStateData> onSuccess,
        System.Action<string>        onError)
    {
        var req = UnityWebRequest.Get($"{BASE_URL}/state");
        yield return req.SendWebRequest();

        if (req.result != UnityWebRequest.Result.Success)
        {
            onError?.Invoke(req.error);
            req.Dispose();
            yield break;
        }

        var data = JsonUtility.FromJson<GameStateData>(req.downloadHandler.text);
        req.Dispose();
        onSuccess?.Invoke(data);
    }

    // ── POST /move ────────────────────────────────────────────────────────
    /// <summary>
    /// Submit a move by flat action index.
    /// Calls onError with the server's error message on HTTP 400 (illegal move).
    /// </summary>
    public IEnumerator PostMove(
        int                          action,
        System.Action<GameStateData> onSuccess,
        System.Action<string>        onError)
    {
        var body = JsonUtility.ToJson(new MoveRequest { action = action });
        var req  = new UnityWebRequest($"{BASE_URL}/move", "POST");
        req.uploadHandler   = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
        req.downloadHandler = new DownloadHandlerBuffer();
        req.SetRequestHeader("Content-Type", "application/json");

        yield return req.SendWebRequest();

        // HTTP 400 → illegal move — recover gracefully
        if (req.responseCode == 400)
        {
            var err = JsonUtility.FromJson<ErrorResponse>(req.downloadHandler.text);
            req.Dispose();
            onError?.Invoke(err?.error ?? "Illegal move");
            yield break;
        }

        if (req.result != UnityWebRequest.Result.Success)
        {
            var msg = req.error;
            req.Dispose();
            onError?.Invoke(msg);
            yield break;
        }

        var data = JsonUtility.FromJson<GameStateData>(req.downloadHandler.text);
        req.Dispose();
        onSuccess?.Invoke(data);
    }

    // ── POST /reset ───────────────────────────────────────────────────────
    /// <summary>Reset the game and return the fresh initial state.</summary>
    public IEnumerator PostReset(
        System.Action<GameStateData> onSuccess,
        System.Action<string>        onError)
    {
        var req = new UnityWebRequest($"{BASE_URL}/reset", "POST");
        req.uploadHandler   = new UploadHandlerRaw(new byte[0]);
        req.downloadHandler = new DownloadHandlerBuffer();
        req.SetRequestHeader("Content-Type", "application/json");

        yield return req.SendWebRequest();

        if (req.result != UnityWebRequest.Result.Success)
        {
            var msg = req.error;
            req.Dispose();
            onError?.Invoke(msg);
            yield break;
        }

        var data = JsonUtility.FromJson<GameStateData>(req.downloadHandler.text);
        req.Dispose();
        onSuccess?.Invoke(data);
    }
}
