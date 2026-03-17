/// <summary>
/// Data classes for JSON deserialisation of Python server responses.
/// Property names match the snake_case keys returned by the server exactly.
/// </summary>
/// 
[System.Serializable]
public class GameStateData
{
    public float[] state_vector;   // 122 floats: [0-120] board, [121] current player
    public int[]   legal_actions;  // flat indices (row*11+col) of empty cells
    public int     current_player; // 1 or 2
    public bool    is_terminal;
    public int     winner;         // 0 = no winner yet, 1 = P1, 2 = P2
}

[System.Serializable]
public class MoveRequest
{
    public int action; // row * 11 + col
}

[System.Serializable]
public class ErrorResponse
{
    public string error;
    public int[]  legal_actions;
}
