using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;

/// <summary>
/// Editor utility that builds the visual 11×11 Hex board in the active scene.
/// Menu: Hex AI ▶ Build Board
/// </summary>
public static class HexBoardSetup
{
    // ── Grid ──────────────────────────────────────────────────────────────
    const int   GridSize  = 11;
    const float HexSize   = 0.50f;  // circumscribed-circle radius (governs cell spacing)
    const float HexVisual = 0.47f;  // rendered radius — slightly smaller creates visible gaps

    // Pointy-top hex layout constants (derived from HexSize)
    static float ColStep  => Mathf.Sqrt(3f) * HexSize;         // Δx between column centres
    static float RowShift => Mathf.Sqrt(3f) * HexSize * 0.5f;  // Δx horizontal shear per row
    static float RowStep  => 1.5f * HexSize;                    // Δy between row centres

    // ── Palette — two muted neutrals + interior ───────────────────────────
    // Player 1 connects top row ↔ bottom row  (terracotta)
    // Player 2 connects left col ↔ right col  (slate blue)
    static readonly Color ColorP1   = new Color(0.78f, 0.42f, 0.38f, 1f);
    static readonly Color ColorP2   = new Color(0.36f, 0.52f, 0.74f, 1f);
    static readonly Color ColorCell = new Color(0.87f, 0.85f, 0.79f, 1f); // warm off-white

    // ── Asset paths ───────────────────────────────────────────────────────
    const string MatFolder   = "Assets/Materials";
    const string MeshFolder  = "Assets/Meshes";
    const string MatP1Path   = MatFolder  + "/HexBorder_P1.mat";
    const string MatP2Path   = MatFolder  + "/HexBorder_P2.mat";
    const string MatCellPath = MatFolder  + "/HexCell.mat";
    const string MeshPath    = MeshFolder + "/HexCell.asset";

    // ── Entry point ───────────────────────────────────────────────────────
    [MenuItem("Hex AI/Build Board")]
    public static void BuildBoard()
    {
        // Remove any previous Board root (undo-safe)
        var old = GameObject.Find("Board");
        if (old != null) Undo.DestroyObjectImmediate(old);

        EnsureFolders();
        EnsureMaterials();

        var hexMesh = GetOrCreateMesh();
        var matP1   = AssetDatabase.LoadAssetAtPath<Material>(MatP1Path);
        var matP2   = AssetDatabase.LoadAssetAtPath<Material>(MatP2Path);
        var matCell = AssetDatabase.LoadAssetAtPath<Material>(MatCellPath);

        // ── Hierarchy ────────────────────────────────────────────────────
        var board = new GameObject("Board");
        Undo.RegisterCreatedObjectUndo(board, "Build Hex Board");

        var cellsRoot = AddChild("Cells", board);
        AddChild("UI", board);

        // ── Cells ─────────────────────────────────────────────────────────
        for (int r = 0; r < GridSize; r++)
        for (int c = 0; c < GridSize; c++)
        {
            var cell = new GameObject($"Cell_{r}_{c}");
            cell.transform.SetParent(cellsRoot.transform, false);
            cell.transform.localPosition = CellPos(r, c);

            cell.AddComponent<MeshFilter>().sharedMesh = hexMesh;
            cell.AddComponent<MeshRenderer>().sharedMaterial =
                PickMaterial(r, c, matP1, matP2, matCell);
        }

        // Centre the board at world origin
        board.transform.position = -BoardCentre();

        ConfigureCamera();
        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        Debug.Log($"[HexBoardSetup] Built {GridSize}×{GridSize} Hex board.");
    }

    // ── Position helpers ──────────────────────────────────────────────────

    // Local position of cell (r, c) relative to Board — parallelogram shear creates
    // the classic Hex board rhombus shape.
    static Vector3 CellPos(int r, int c) =>
        new Vector3(c * ColStep + r * RowShift, -r * RowStep, 0f);

    // Geometric centre of the full grid in local space
    static Vector3 BoardCentre() =>
        (CellPos(0, 0) + CellPos(GridSize - 1, GridSize - 1)) * 0.5f;

    // ── Material selection ────────────────────────────────────────────────
    static Material PickMaterial(int r, int c, Material p1, Material p2, Material neutral)
    {
        if (r == 0 || r == GridSize - 1) return p1;  // P1 top / bottom border rows
        if (c == 0 || c == GridSize - 1) return p2;  // P2 left / right border columns
        return neutral;
    }

    // ── Camera ────────────────────────────────────────────────────────────
    static void ConfigureCamera()
    {
        var cam = Camera.main;
        if (cam == null) return;

        cam.orthographic       = true;
        cam.clearFlags         = CameraClearFlags.SolidColor;
        cam.backgroundColor    = new Color(0.13f, 0.14f, 0.16f);
        cam.transform.position = new Vector3(0f, 0f, -10f);
        cam.transform.rotation = Quaternion.identity;

        // Fit the board vertically with padding; width adjusts with screen aspect ratio
        float boardH = (GridSize - 1) * RowStep + HexSize * 2f;
        cam.orthographicSize = boardH * 0.5f + 1f;
    }

    // ── Mesh ─────────────────────────────────────────────────────────────
    static Mesh GetOrCreateMesh()
    {
        var existing = AssetDatabase.LoadAssetAtPath<Mesh>(MeshPath);
        if (existing != null) return existing;

        var mesh = BuildHexMesh();
        AssetDatabase.CreateAsset(mesh, MeshPath);
        AssetDatabase.SaveAssets();
        return mesh;
    }

    // Pointy-top hexagon centred at origin.
    // Vertices are wound clockwise (viewed from +Z) so the front face points
    // toward the camera at z = −10.
    static Mesh BuildHexMesh()
    {
        var mesh  = new Mesh { name = "HexCell" };
        var verts = new Vector3[7]; // [0] centre, [1..6] perimeter corners
        verts[0]  = Vector3.zero;

        for (int i = 0; i < 6; i++)
        {
            // 30° start, step clockwise (decreasing angle) for correct winding
            float a = Mathf.Deg2Rad * (30f - 60f * i);
            verts[i + 1] = new Vector3(
                HexVisual * Mathf.Cos(a),
                HexVisual * Mathf.Sin(a),
                0f);
        }

        mesh.vertices  = verts;
        // Fan triangulation from centre: six triangles covering the hexagon
        mesh.triangles = new int[] { 0,1,2, 0,2,3, 0,3,4, 0,4,5, 0,5,6, 0,6,1 };
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    // ── Asset management ──────────────────────────────────────────────────
    static GameObject AddChild(string name, GameObject parent)
    {
        var go = new GameObject(name);
        go.transform.SetParent(parent.transform, false);
        return go;
    }

    static void EnsureFolders()
    {
        if (!AssetDatabase.IsValidFolder(MatFolder))
            AssetDatabase.CreateFolder("Assets", "Materials");
        if (!AssetDatabase.IsValidFolder(MeshFolder))
            AssetDatabase.CreateFolder("Assets", "Meshes");
    }

    static void EnsureMaterials()
    {
        EnsureMaterial(MatP1Path,   ColorP1);
        EnsureMaterial(MatP2Path,   ColorP2);
        EnsureMaterial(MatCellPath, ColorCell);
    }

    static void EnsureMaterial(string path, Color color)
    {
        if (AssetDatabase.LoadAssetAtPath<Material>(path) != null) return;

        // URP Unlit preferred; falls back to built-in Unlit/Color if not found
        var shader = Shader.Find("Universal Render Pipeline/Unlit")
                  ?? Shader.Find("Unlit/Color");
        var mat = new Material(shader);
        mat.SetColor("_BaseColor", color);
        AssetDatabase.CreateAsset(mat, path);
    }
}
