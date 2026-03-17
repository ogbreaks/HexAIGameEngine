using System.Collections.Generic;
using UnityEngine;
using Unity.InferenceEngine;


/// <summary>
/// Loads a trained ONNX model and runs inference to pick the best legal action.
///
/// Uses Unity Sentis (NOT Barracuda — deprecated in Unity 6.x).
/// All inference runs synchronously on the CPU so it never blocks a frame for
/// longer than a few milliseconds on an 11×11 board.
///
/// Assign ONNX ModelAssets in the Inspector, then call LoadModel() once to
/// activate a difficulty level before the game starts.
///
/// Network I/O contract (matches ML-Agents PPO export):
///   Input  "obs_0"             — shape [1, 122]  (state vector)
///   Output "discrete_actions"  — shape [1, 121]  (action logits / log-probs)
///
/// If the output layer name differs from your export, set OutputLayerName in
/// the Inspector to match.
/// </summary>
public class AIManager : MonoBehaviour
{
    // ── Inspector ──────────────────────────────────────────────────────────
    [Header("ONNX Models — assign after training")]
    public ModelAsset easyModel;
    public ModelAsset mediumModel;
    public ModelAsset hardModel;
    public ModelAsset expertModel;

    [Header("Inference settings")]
    [Tooltip("Output layer name exported by mlagents-learn. " +
             "Typical value: 'discrete_actions'. Check with Netron if unsure.")]
    public string outputLayerName = "discrete_actions";

    // ── Private state ──────────────────────────────────────────────────────
    private Worker _worker;
    private Model   _runtimeModel;
    private Difficulty _currentDifficulty = Difficulty.Medium;

    // ── Difficulty enum ────────────────────────────────────────────────────
    public enum Difficulty { Easy, Medium, Hard, Expert }

    public Difficulty CurrentDifficulty => _currentDifficulty;

    // ── Public API ─────────────────────────────────────────────────────────

    /// <summary>
    /// Dispose the currently loaded model and load the one for the requested
    /// difficulty.  Safe to call at any time including mid-game.
    /// </summary>
    public void LoadModel(Difficulty difficulty)
    {
        _currentDifficulty = difficulty;

        _worker?.Dispose();
        _worker = null;

        var asset = difficulty switch
        {
            Difficulty.Easy   => easyModel,
            Difficulty.Medium => mediumModel,
            Difficulty.Hard   => hardModel,
            Difficulty.Expert => expertModel,
            _                 => mediumModel
        };

        if (asset == null)
        {
            Debug.LogWarning(
                $"[AIManager] No ONNX model assigned for {difficulty}. " +
                "AI will play random legal moves until a model is provided.");
            return;
        }

        _runtimeModel = ModelLoader.Load(asset);
        _worker       = new Worker(_runtimeModel, BackendType.CPU);
        Debug.Log($"[AIManager] {difficulty} model loaded.");
    }

    /// <summary>
    /// Run inference on the given state vector and return the index of the
    /// best legal action according to the model.
    ///
    /// Falls back to a random legal action when no model is loaded so the
    /// game remains playable before training is complete.
    /// </summary>
    public int GetBestAction(float[] stateVector, List<int> legalActions)
    {
        if (legalActions == null || legalActions.Count == 0) return 0;

        if (_worker == null)
            return legalActions[UnityEngine.Random.Range(0, legalActions.Count)];

        // ── Build input tensor ─────────────────────────────────────────────
        using var inputTensor = new Tensor<float>(new TensorShape(1, 122), stateVector);

        // ── Run inference ──────────────────────────────────────────────────
        _worker.SetInput("obs_0", inputTensor);
        _worker.Schedule();

        // ── Read output ────────────────────────────────────────────────────
        var outputTensor = _worker.PeekOutput(outputLayerName) as Tensor<float>;
        if (outputTensor == null)
        {
            // Fallback: try the default (first) output
            outputTensor = _worker.PeekOutput() as Tensor<float>;
        }

        if (outputTensor == null)
        {
            Debug.LogWarning("[AIManager] Could not read output tensor. Falling back to random.");
            return legalActions[UnityEngine.Random.Range(0, legalActions.Count)];
        }

        outputTensor.DownloadToArray(); // sync to CPU

        // ── Argmax over legal actions ──────────────────────────────────────
        // The tensor is logits/log-probs of shape [1, 121].
        int   bestAction = legalActions[0];
        float bestScore  = float.NegativeInfinity;

        foreach (int a in legalActions)
        {
            float score = outputTensor[a];
            if (score > bestScore)
            {
                bestScore  = score;
                bestAction = a;
            }
        }

        return bestAction;
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────

    private void OnDestroy()
    {
        _worker?.Dispose();
    }
}
