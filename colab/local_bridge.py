"""Local Bridge Proxy for YuE2 Studio.

Forwards local engine requests from http://127.0.0.1:18087 to the
remote Google Colab tunnel URL (e.g. Cloudflare / ngrok).
Enables the official pre-built YuE2 Studio desktop application to run
inference directly on Google Colab Tesla T4 without a local NVIDIA GPU.
"""

from __future__ import annotations

import argparse
import http.server
import logging
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("yue2-bridge")

CHUNK_SIZE = 64 * 1024
is_remote_healthy = False


def health_monitor(remote_url: str) -> None:
    """Continuously monitors remote engine health in a background thread."""
    global is_remote_healthy
    first_check = True

    while True:
        healthy_now = False
        for ep in ("/health", "/props"):
            try:
                req = urllib.request.Request(
                    f"{remote_url.rstrip('/')}{ep}", headers={"User-Agent": "YuE2-Bridge"}
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        healthy_now = True
                        break
            except Exception:
                pass

        if healthy_now:
            if not is_remote_healthy:
                logger.info("Remote engine is HEALTHY (200 OK)!")
            is_remote_healthy = True
        else:
            if is_remote_healthy or first_check:
                logger.warning("Remote engine health check waiting...")
            is_remote_healthy = False

        first_check = False
        time.sleep(2)


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    """Transparent HTTP reverse proxy handler with fast-path local health response."""

    remote_url: str = ""

    def do_GET(self) -> None:
        self._proxy_request("GET")

    def do_POST(self) -> None:
        self._proxy_request("POST")

    def do_HEAD(self) -> None:
        self._proxy_request("HEAD")

    def _proxy_request(self, method: str) -> None:
        # Fast path for YuE2 Studio local health check (which has a strict 300ms socket timeout)
        if self.path == "/health" and method == "GET":
            try:
                if is_remote_healthy:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", "2")
                    self.end_headers()
                    self.wfile.write(b"ok")
                    return
                else:
                    self.send_response(503)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", "11")
                    self.end_headers()
                    self.wfile.write(b"unavailable")
                    return
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                return

        target = f"{self.remote_url.rstrip('/')}{self.path}"
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        req_headers: dict[str, str] = {}
        for header, value in self.headers.items():
            if header.lower() not in ("host", "content-length"):
                req_headers[header] = value
        if getattr(self, "engine_name", ""):
            req_headers["X-Engine"] = self.engine_name

        req = urllib.request.Request(
            target,
            data=body,
            headers=req_headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                for header, value in resp.headers.items():
                    if header.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(header, value)
                resp_data = resp.read()
                self.send_header("Content-Length", str(len(resp_data)))
                self.end_headers()
                self.wfile.write(resp_data)
        except urllib.error.HTTPError as exc:
            try:
                self.send_response(exc.code)
                self.end_headers()
                self.wfile.write(exc.read())
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass
        except Exception as exc:  # noqa: BLE001
            logger.error("Proxy error to %s: %s", target, exc)
            try:
                self.send_response(502)
                self.end_headers()
                self.wfile.write(f"Bridge proxy error: {exc}".encode())
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass

    def log_message(self, format: str, *args: object) -> None:
        # Don't flood console with periodic /health checks
        if args and str(args[0]).startswith("GET /health"):
            return
        logger.info("%s - %s", self.address_string(), format % args)


def extract_url(text: str) -> str | None:
    """Extracts a valid http/https URL from user input text."""
    match = re.search(r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z0-9.-]+(?:/[^\s]*)?", text)
    if match:
        return match.group(0).rstrip("/")
    return None


def launch_desktop_app(app_choice: str = "none") -> None:
    """Finds and launches YuE2 Studio and/or ACE-Step Studio desktop application."""
    if app_choice == "none":
        return
    project_root = Path(__file__).resolve().parent.parent

    yue_candidates = [
        project_root / "YuE2-Studio-3.4.0-portable-windows-x64" / "YuE2-Studio.exe",
        project_root / "YuE2-Studio.exe",
    ]
    ace_candidates = [
        project_root / "ACE-Step-Studio-3.4.0-portable-windows-x64" / "ACE-Step-Studio.exe",
        project_root / "ACE-Step-Studio.exe",
    ]
    minimax_candidates = [
        project_root / "MiniMax-Music3-Studio-3.4.0-portable" / "MiniMax-Music3-Studio.exe",
        project_root / "MiniMax-Music3-Studio-3.4.0-portable-windows-x64" / "MiniMax-Music3-Studio.exe",
        project_root / "MiniMax-Music3-Studio.exe",
    ]

    to_launch: list[tuple[str, Path]] = []
    if app_choice in ("yue2", "auto"):
        for c in yue_candidates:
            if c.exists():
                to_launch.append(("YuE2 Studio", c))
                break
    if app_choice in ("acestep",):
        for c in ace_candidates:
            if c.exists():
                to_launch.append(("ACE-Step Studio", c))
                break
    if app_choice in ("minimax",):
        for c in minimax_candidates:
            if c.exists():
                to_launch.append(("MiniMax Music3 Studio", c))
                break
    if app_choice == "both":
        for c in yue_candidates:
            if c.exists():
                to_launch.append(("YuE2 Studio", c))
                break
        for c in ace_candidates:
            if c.exists():
                to_launch.append(("ACE-Step Studio", c))
                break

    for name, exe in to_launch:
        logger.info("Launching %s desktop application: %s", name, exe)
        try:
            env = os.environ.copy()
            if "YuE2" in name:
                env["YUE_ENGINE_BASE_URL"] = "http://127.0.0.1:18087"
                env["YUE_ENGINE_URL"] = "http://127.0.0.1:18087"
                env["YUE_ENGINE_PORT"] = "18087"
            elif "ACE" in name:
                env["STUDIO_ENGINE_URL"] = "http://127.0.0.1:8089"
                env["STUDIO_ENGINE_PORT"] = "8089"
            elif "MiniMax" in name:
                env["MINIMAX_MUSIC_CPP_BASE_URL"] = "http://127.0.0.1:8086"
                env["MINIMAX_MM_SERVER_PORT"] = "8086"
            subprocess.Popen([str(exe)], cwd=str(exe.parent), env=env)
        except Exception as exc:
            logger.warning("Could not launch %s automatically: %s", name, exc)


def main() -> None:
    """CLI entrypoint."""
    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Local bridge proxy for YuE2 Studio -> Google Colab"
    )
    parser.add_argument(
        "remote_url",
        nargs="?",
        default=None,
        type=str,
        help="Public Cloudflare/ngrok tunnel URL (e.g. https://xxx.trycloudflare.com)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=18087,
        help="Local port to listen on (default: 18087)",
    )
    parser.add_argument(
        "--app",
        choices=["yue2", "acestep", "minimax", "both", "none", "auto"],
        default="none",
        help="Desktop application to launch: yue2, acestep, minimax, both, none (default: none)",
    )

    args = parser.parse_args()
    raw_url = args.remote_url

    if not raw_url:
        print("=" * 65)
        print("   YuE2 / ACE-Step / MiniMax Studio — Мост подключения к Kaggle / Colab")
        print("=" * 65)
        print("\nВставьте ссылку из блокнота Kaggle или Google Colab:")
        print("(например: https://xxxx-xxxx-xxxx.trycloudflare.com или ngrok)")
        while not raw_url:
            user_input = input("\nСсылка из Kaggle/Colab > ").strip()
            found = extract_url(user_input)
            if found:
                raw_url = found
                break
            print("[!] В тексте не найдена ссылка (https://...). Попробуйте снова.")

    remote = extract_url(raw_url) or raw_url.strip()
    if not remote.startswith("http://") and not remote.startswith("https://"):
        remote = f"https://{remote}"

    logger.info("Connecting local port %d -> %s", args.port, remote)

    class YueHandler(ProxyHandler):
        engine_name = "yue2"

    class AceHandler(ProxyHandler):
        engine_name = "acestep"

    class MiniMaxHandler(ProxyHandler):
        engine_name = "minimax"

    YueHandler.remote_url = remote
    AceHandler.remote_url = remote
    MiniMaxHandler.remote_url = remote

    # Start health monitor background thread
    monitor_thread = threading.Thread(
        target=health_monitor, args=(remote,), daemon=True
    )
    monitor_thread.start()

    try:
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), YueHandler)
    except OSError as exc:
        logger.error(
            "Could not bind to 127.0.0.1:%d (%s).",
            args.port,
            exc,
        )
        sys.exit(1)

    httpd_ace = None
    try:
        httpd_ace = http.server.ThreadingHTTPServer(("127.0.0.1", 8089), AceHandler)
        threading.Thread(target=httpd_ace.serve_forever, daemon=True).start()
    except OSError:
        pass

    httpd_minimax = None
    try:
        httpd_minimax = http.server.ThreadingHTTPServer(("127.0.0.1", 8086), MiniMaxHandler)
        threading.Thread(target=httpd_minimax.serve_forever, daemon=True).start()
    except OSError:
        pass

    logger.info("=" * 60)
    logger.info("🌉 YuE2, ACE-Step & MiniMax Local Bridge is ACTIVE!")
    logger.info("YuE2 Port:     http://127.0.0.1:%d", args.port)
    logger.info("ACE-Step Port: http://127.0.0.1:8089")
    logger.info("MiniMax Port:  http://127.0.0.1:8086")
    logger.info("Forwarding to: %s", remote)
    logger.info("=" * 60)

    launch_desktop_app(args.app)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Bridge stopped.")
    finally:
        httpd.server_close()
        if httpd_ace:
            httpd_ace.server_close()
        if httpd_minimax:
            httpd_minimax.server_close()


if __name__ == "__main__":
    main()
