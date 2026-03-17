"""
test_hex_server.py — integration tests for the HTTP endpoints.

Uses http.server directly via a background thread; no external
test client libraries required (pytest + stdlib only).
"""

from __future__ import annotations

import http.client
import json
import threading
import urllib.parse
from http.server import HTTPServer

import pytest

# Import module-level state so we can reset between tests
import server.hex_server as hex_server_module
from server.game_manager import GameManager
from server.hex_server import make_server

SIZE = 11
TOTAL_CELLS = SIZE * SIZE

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def server_url():
    """Start a test server on a free port for the entire test module."""
    httpd: HTTPServer = make_server("localhost", 0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://localhost:{port}"
    httpd.shutdown()


@pytest.fixture(autouse=True)
def reset_game_state():
    """Reset the shared game manager before every test."""
    hex_server_module._manager = GameManager()
    yield


# ── Helpers ───────────────────────────────────────────────────────────────
# http.client is used directly so we avoid per-request TCP handshake
# overhead that urllib.request incurs (significant on Windows).


def _conn(url: str) -> tuple[http.client.HTTPConnection, str]:
    parsed = urllib.parse.urlparse(url)
    return http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10), (
        parsed.path or "/"
    )


def _get(url: str) -> tuple[int, dict]:
    conn, path = _conn(url)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def _post(url: str, payload: dict) -> tuple[int, dict]:
    conn, path = _conn(url)
    body = json.dumps(payload).encode()
    try:
        conn.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


_REQUIRED_FIELDS = {
    "state_vector",
    "legal_actions",
    "current_player",
    "is_terminal",
    "winner",
}


# ── GET /state ────────────────────────────────────────────────────────────


def test_get_state_returns_200(server_url):
    status, _ = _get(f"{server_url}/state")
    assert status == 200


def test_get_state_returns_all_required_fields(server_url):
    _, body = _get(f"{server_url}/state")
    assert _REQUIRED_FIELDS.issubset(body.keys())


def test_get_state_state_vector_length_is_122(server_url):
    _, body = _get(f"{server_url}/state")
    assert len(body["state_vector"]) == 122


def test_get_state_legal_actions_length_is_121_on_fresh_game(server_url):
    _, body = _get(f"{server_url}/state")
    assert len(body["legal_actions"]) == TOTAL_CELLS


def test_get_state_current_player_is_1_on_fresh_game(server_url):
    _, body = _get(f"{server_url}/state")
    assert body["current_player"] == 1


def test_get_state_is_terminal_false_on_fresh_game(server_url):
    _, body = _get(f"{server_url}/state")
    assert body["is_terminal"] is False


def test_get_state_winner_is_null_on_fresh_game(server_url):
    _, body = _get(f"{server_url}/state")
    assert body["winner"] is None


def test_get_unknown_endpoint_returns_404(server_url):
    status, _ = _get(f"{server_url}/unknown")
    assert status == 404


# ── POST /move ────────────────────────────────────────────────────────────


def test_post_move_valid_action_returns_200(server_url):
    status, _ = _post(f"{server_url}/move", {"action": 60})
    assert status == 200


def test_post_move_valid_action_returns_all_required_fields(server_url):
    _, body = _post(f"{server_url}/move", {"action": 60})
    assert _REQUIRED_FIELDS.issubset(body.keys())


def test_post_move_reduces_legal_actions_by_1(server_url):
    _, body = _post(f"{server_url}/move", {"action": 60})
    assert len(body["legal_actions"]) == TOTAL_CELLS - 1


def test_post_move_switches_current_player(server_url):
    _, body = _post(f"{server_url}/move", {"action": 60})
    assert body["current_player"] == 2


def test_post_move_played_cell_not_in_legal_actions(server_url):
    _, body = _post(f"{server_url}/move", {"action": 60})
    assert 60 not in body["legal_actions"]


def test_post_move_illegal_action_returns_400(server_url):
    status, _ = _post(f"{server_url}/move", {"action": TOTAL_CELLS + 99})
    assert status == 400


def test_post_move_occupied_cell_returns_400(server_url):
    _post(f"{server_url}/move", {"action": 0})  # P1
    _post(f"{server_url}/move", {"action": 1})  # P2
    status, _ = _post(f"{server_url}/move", {"action": 0})  # occupied
    assert status == 400


def test_post_move_400_response_contains_error_field(server_url):
    status, body = _post(f"{server_url}/move", {"action": TOTAL_CELLS + 1})
    assert status == 400
    assert "error" in body


def test_post_move_400_response_contains_legal_actions(server_url):
    status, body = _post(f"{server_url}/move", {"action": TOTAL_CELLS + 1})
    assert status == 400
    assert "legal_actions" in body


def test_post_move_missing_action_field_returns_400(server_url):
    status, _ = _post(f"{server_url}/move", {"not_action": 60})
    assert status == 400


def test_post_unknown_endpoint_returns_404(server_url):
    status, _ = _post(f"{server_url}/unknown", {})
    assert status == 404


# ── POST /reset ───────────────────────────────────────────────────────────


def test_post_reset_returns_200(server_url):
    status, _ = _post(f"{server_url}/reset", {})
    assert status == 200


def test_post_reset_returns_fresh_state(server_url):
    _post(f"{server_url}/move", {"action": 0})
    _, body = _post(f"{server_url}/reset", {})
    assert len(body["legal_actions"]) == TOTAL_CELLS


def test_post_reset_returns_all_required_fields(server_url):
    _, body = _post(f"{server_url}/reset", {})
    assert _REQUIRED_FIELDS.issubset(body.keys())


def test_post_reset_clears_winner(server_url):
    _, body = _post(f"{server_url}/reset", {})
    assert body["winner"] is None


def test_post_reset_current_player_is_1(server_url):
    _post(f"{server_url}/move", {"action": 0})
    _, body = _post(f"{server_url}/reset", {})
    assert body["current_player"] == 1


# ── Content-Type ──────────────────────────────────────────────────────────


def test_get_state_content_type_is_json(server_url):
    conn, path = _conn(f"{server_url}/state")
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        resp.read()
        assert "application/json" in resp.headers.get("Content-Type", "")
    finally:
        conn.close()


def test_post_move_content_type_is_json(server_url):
    conn, path = _conn(f"{server_url}/move")
    body = json.dumps({"action": 5}).encode()
    try:
        conn.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        resp = conn.getresponse()
        resp.read()
        assert "application/json" in resp.headers.get("Content-Type", "")
    finally:
        conn.close()


# ── Full game via HTTP ─────────────────────────────────────────────────────


def test_full_game_completes_via_http(server_url):
    """Play a complete game action-by-action through the server endpoints.

    Uses a single persistent HTTP/1.1 connection to avoid per-move TCP
    handshake overhead.
    """
    import random

    parsed = urllib.parse.urlparse(server_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)

    def _req_get(path: str) -> dict:
        conn.request("GET", path)
        r = conn.getresponse()
        return json.loads(r.read())

    def _req_post(path: str, payload: dict) -> tuple[int, dict]:
        body = json.dumps(payload).encode()
        conn.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        r = conn.getresponse()
        return r.status, json.loads(r.read())

    try:
        _req_post("/reset", {})
        state = _req_get("/state")

        while not state["is_terminal"]:
            action = random.choice(state["legal_actions"])
            status, state = _req_post("/move", {"action": action})
            assert status == 200
    finally:
        conn.close()

    assert state["is_terminal"] is True
    assert state["winner"] in (1, 2)
    assert state["legal_actions"] == []
