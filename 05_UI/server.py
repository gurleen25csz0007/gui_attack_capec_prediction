"""HTTP server for the CVE-ID/description-to-CAPEC interface."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from predictor import CAPECPredictor, ModelNotConfiguredError


APP_DIR = Path(__file__).resolve().parent
PREDICTOR: CAPECPredictor | None = None


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "CAPECUI/2.0"

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        request_path = urlparse(self.path).path
        if request_path == "/health":
            self._json(PREDICTOR.status() if PREDICTOR else {"status": "loading"})
            return

        relative_path = "index.html" if request_path == "/" else request_path.lstrip("/")
        file_path = (APP_DIR / relative_path).resolve()
        if APP_DIR not in file_path.parents or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        body = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 1_000_000:
                raise ValueError("Invalid request size.")

            payload = json.loads(self.rfile.read(content_length))
            text = payload.get("input", payload.get("text", ""))
            top_k = payload.get("top_k", 10)
            if PREDICTOR is None:
                raise ModelNotConfiguredError("The CAPEC model is still loading.")
            self._json(PREDICTOR.predict(text=text, top_k=top_k))
        except LookupError as exc:
            self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except ModelNotConfiguredError as exc:
            self._json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            print(f"Prediction error: {exc}", flush=True)
            self._json(
                {"error": "Prediction failed. Check the server terminal for details."},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CVE/description-to-CAPEC UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global PREDICTOR
    print("Loading our-approach fusion and dataset index...", flush=True)
    PREDICTOR = CAPECPredictor()
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"Ready: http://{args.host}:{args.port}", flush=True)
    if not PREDICTOR.ready:
        print(f"Model unavailable: {PREDICTOR.status()['message']}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
