"""
game_manager.py — manages a single active HexGame instance.

Imported by the HTTP server. The server holds one GameManager; all
incoming requests mutate or inspect it.
"""

from __future__ import annotations

from game.hex_actions import get_legal_actions
from game.hex_game import HexGame
from game.hex_state_vector import get_state_vector


class GameManager:
    """Single-game state container with a clean dict-based API."""

    def __init__(self) -> None:
        self.game: HexGame = HexGame()

    # ── Public API ──────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """
        Return the complete current game state as a dictionary.

        Keys
        ----
        state_vector   : list[float]  — 122 floats (see hex_state_vector.py)
        legal_actions  : list[int]    — indices of empty cells (empty when terminal)
        current_player : int          — 1 or 2
        is_terminal    : bool
        winner         : int | None   — 1, 2, or None
        """
        return self._build_state_dict()

    def apply_move(self, action: int) -> dict:
        """
        Validate and apply *action*; return the updated state dict.

        Raises
        ------
        ValueError
            If the action is not in the current legal-action list
            (covers: out of range, occupied cell, game already terminal).
        """
        legal = get_legal_actions(self.game)
        if action not in legal:
            raise ValueError(
                f"Action {action} is not legal. "
                f"Legal actions: {legal[:10]}{'...' if len(legal) > 10 else ''}"
            )
        self.game = self.game.apply_action(action)
        return self._build_state_dict()

    def reset(self) -> dict:
        """Start a new game and return the initial state dict."""
        self.game = HexGame()
        return self._build_state_dict()

    # ── Internal ────────────────────────────────────────────────────────────

    def _build_state_dict(self) -> dict:
        return {
            "state_vector": get_state_vector(self.game),
            "legal_actions": get_legal_actions(self.game),
            "current_player": self.game.get_current_player(),
            "is_terminal": self.game.is_terminal(),
            "winner": self.game.get_winner(),
        }
