"""
hex_server.py — lightweight HTTP server (stdlib only, no Flask/FastAPI).

Endpoints
---------
GET  /state   — current game state
POST /move    — apply a move  { "action": N }
POST /reset   — start a new game

All responses are JSON with Content-Type: application/json.
CORS headers are included so Unity WebGL / UnityWebRequest can call freely.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTP server that handles each request in a separate thread."""

    daemon_threads = True


from server.game_manager import GameManager

HOST = "localhost"
PORT = 5000

# Module-level game manager shared across all requests.
# _manager_lock guards all reads and writes from handler threads.
_manager = GameManager()
_manager_lock = threading.Lock()

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


class HexRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Hex AI game server."""

    # HTTP/1.0: one request per connection — simpler and avoids keep-alive
    # connection-reset issues with Unity's UnityWebRequest/libcurl.
    # ThreadingMixIn gives us concurrency so performance is unaffected.
    protocol_version = "HTTP/1.0"
    # ── Routing ─────────────────────────────────────────────────────────────

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Pre-flight CORS request."""
        self._send_response(200, {})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/state":
            with _manager_lock:
                state = _manager.get_state()
            self._send_response(200, state)
        else:
            self._send_response(404, {"error": f"Unknown endpoint: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        body = self._read_body()

        if self.path == "/move":
            self._handle_move(body)
        elif self.path == "/reset":
            with _manager_lock:
                result = _manager.reset()
            self._send_response(200, result)
        else:
            self._send_response(404, {"error": f"Unknown endpoint: {self.path}"})

    # ── Handlers ─────────────────────────────────────────────────────────────

    def _handle_move(self, body: dict) -> None:
        if "action" not in body:
            self._send_response(
                400, {"error": "Missing 'action' field in request body."}
            )
            return

        action = body["action"]
        if not isinstance(action, int):
            self._send_response(
                400,
                {"error": f"'action' must be an integer, got {type(action).__name__}."},
            )
            return

        try:
            with _manager_lock:
                state = _manager.apply_move(action)
            self._send_response(200, state)
        except ValueError:
            with _manager_lock:
                legal = _manager.get_state()["legal_actions"]
            self._send_response(
                400,
                {
                    "error": "Illegal move",
                    "legal_actions": legal,
                },
            )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _send_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for header, value in _CORS_HEADERS.items():
            self.send_header(header, value)
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def log_message(self, fmt: str, *args) -> None:  # noqa: ANN002
        """Suppress default Apache-style access logs."""


# ── Public API used by main.py ───────────────────────────────────────────────


def make_server(host: str = HOST, port: int = PORT) -> ThreadingHTTPServer:
    """Create and return (but do not start) the HTTP server."""
    return ThreadingHTTPServer((host, port), HexRequestHandler)


def run_server(host: str = HOST, port: int = PORT) -> None:
    """Start the server and block until interrupted."""
    server = make_server(host, port)
    print(f"Hex AI server running on {host}:{port}")
    print(f"GET  /state  — get current game state")
    print(f"POST /move   — apply move {{action: N}}")
    print(f"POST /reset  — reset game")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()
