"""Serve the local demo web app and its auth API (see ``demo_backend``).

    python -m qa_agent.tools.demo_server            # http://127.0.0.1:8765
    python -m qa_agent.tools.demo_server --port 9000 --dir demo_app --db data/demo_users.db

Then set UI_TEST_BASE_URL and API_TEST_BASE_URL to http://127.0.0.1:8765.
Binds to loopback by default; in a container pass ``--host 0.0.0.0`` (Docker Compose
publishes the port on the host's loopback only).
"""

import argparse
import functools
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from qa_agent.tools.demo_backend import ApiResponse, DemoAuthService

DEFAULT_DEMO_DIR = Path("demo_app")
DEFAULT_DEMO_DB = Path("data/demo_users.db")
_MAX_BODY_BYTES = 64 * 1024


class _DemoHandler(SimpleHTTPRequestHandler):
    """Static files from the demo directory; ``/api/...`` goes to the auth service."""

    def __init__(self, *args: object, service: DemoAuthService, **kwargs: object) -> None:
        self.service = service
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        pass

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if not url.path.startswith("/api/"):
            super().do_GET()
        elif url.path == "/api/v2/me":
            self._send(self.service.me(self._bearer()))
        elif url.path == "/api/demo/outbox":
            self._send(self.service.outbox_for(parse_qs(url.query).get("email", [""])[0]))
        else:
            self._send(ApiResponse(404, {"error": "not_found"}))

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        routes = {
            "/api/v2/register": self.service.register,
            "/api/v2/login": self.service.login,
            "/api/v2/password-reset": self.service.request_password_reset,
            "/api/v2/password-reset/confirm": self.service.confirm_password_reset,
        }
        if path == "/api/v2/logout":
            self._send(self.service.logout(self._bearer()))
        elif path in routes:
            body = self._json_body()
            if body is None:
                self._send(ApiResponse(400, {"error": "invalid_json"}))
            else:
                self._send(routes[path](body))
        else:
            self._send(ApiResponse(404 if path.startswith("/api/") else 501, {"error": "not_found"}))

    def _bearer(self) -> str | None:
        auth = self.headers.get("Authorization", "")
        return auth[7:].strip() or None if auth.lower().startswith("bearer ") else None

    def _json_body(self) -> dict | None:
        try:
            length = min(int(self.headers.get("Content-Length") or 0), _MAX_BODY_BYTES)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, UnicodeDecodeError):
            return None
        return body if isinstance(body, dict) else None

    def _send(self, response: ApiResponse) -> None:
        payload = b"" if response.body is None else json.dumps(response.body).encode()
        self.send_response(response.status)
        if response.body is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        for name, value in response.headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)


def _server(
    directory: Path, port: int, db_path: str = ":memory:", host: str = "127.0.0.1", public_url: str = ""
) -> ThreadingHTTPServer:
    if not directory.is_dir():
        raise FileNotFoundError(f"Demo app directory not found: {directory}")
    service = DemoAuthService(db_path)
    handler = functools.partial(_DemoHandler, directory=str(directory), service=service)
    server = ThreadingHTTPServer((host, port), handler)
    bound_host, bound_port = server.server_address[:2]
    if bound_host in ("0.0.0.0", "::"):  # links in reset emails must be openable in a browser
        bound_host = "127.0.0.1"
    service.base_url = public_url.rstrip("/") or f"http://{bound_host!s}:{bound_port}"
    server.demo_service = service  # type: ignore[attr-defined]
    return server


@contextmanager
def serve_demo_app(
    directory: str | Path = DEFAULT_DEMO_DIR, port: int = 0, db_path: str = ":memory:"
) -> Iterator[str]:
    """Serve ``directory`` and the demo API on loopback in a background thread; yields the
    base URL. Users are kept in memory unless ``db_path`` names a SQLite file."""
    server = _server(Path(directory), port, db_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, bound_port = server.server_address[:2]
        yield f"http://{host}:{bound_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=str(DEFAULT_DEMO_DIR))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", default=str(DEFAULT_DEMO_DB), help="SQLite file for demo users")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (0.0.0.0 inside a container)")
    parser.add_argument(
        "--public-url", default="", help="Base URL used in reset-email links (default: the bound address)"
    )
    args = parser.parse_args()
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    server = _server(Path(args.dir), args.port, args.db, args.host, args.public_url)
    base_url = server.demo_service.base_url  # type: ignore[attr-defined]
    print(f"Serving {args.dir} and the demo API at {base_url} (bound to {args.host}; Ctrl+C to stop)", flush=True)
    print("Demo account: registered.user@example.com / N3w-Passw0rd!", flush=True)
    print(f"Reset emails (demo inbox): {base_url}/outbox.html", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
