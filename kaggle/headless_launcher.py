"""YuE2 & ACE-Step Studio — Headless Kaggle Cloud Launcher.

Automates the background launch of Kaggle Tesla T4 GPU engines:
1. Authenticates with Kaggle API and checks kernel status.
2. Starts kernel only if not already running.
3. Listens for Cloudflare tunnel URL announcement via ntfy without blocking.
4. Saves active tunnel URL for desktop studios.
5. Launches desktop application.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Patch open() for Kaggle API on Windows (missing utf-8 encoding in kernels_push)
_orig_open = builtins.open

def _utf8_open(*args, **kwargs):
    if len(args) > 1 and "b" in str(args[1]):
        return _orig_open(*args, **kwargs)
    if "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _orig_open(*args, **kwargs)

builtins.open = _utf8_open

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kaggle-launcher")

DEFAULT_USERNAME = "jungterdai"


def compute_ntfy_topic(username: str, app: str = "yue2") -> str:
    """Derives a deterministic, private ntfy topic from username and app."""
    h = hashlib.sha256(username.strip().lower().encode("utf-8")).hexdigest()[:16]
    prefixes = {"acestep": "acestep_studio"}
    prefix = prefixes.get(app, "yue2_studio")
    return f"{prefix}_{h}"


def ensure_kaggle_credentials() -> str:
    """Ensures Kaggle credentials are configured locally and returns username."""
    kdir = Path.home() / ".kaggle"
    kdir.mkdir(parents=True, exist_ok=True)
    json_file = kdir / "kaggle.json"
    username = DEFAULT_USERNAME
    if json_file.exists():
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            username = data.get("username", username)
        except Exception:
            pass
    os.environ["KAGGLE_USERNAME"] = username
    return username


def verify_tunnel_health(url: str, timeout: float = 2.5) -> bool:
    """Verifies that a tunnel candidate answers /health or /props with HTTP 200."""
    clean = url.strip().rstrip("/")
    if not (clean.startswith("http://") or clean.startswith("https://")):
        return False
    probe_headers = {
        "User-Agent": "YueProbe",
        "ngrok-skip-browser-warning": "true",
    }
    try:
        req = urllib.request.Request(f"{clean}/health", headers=probe_headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass
    try:
        req = urllib.request.Request(f"{clean}/props", headers=probe_headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass
    return False


def get_cached_tunnel_url() -> str | None:
    """Checks whether a previously saved tunnel URL is still active and healthy."""
    project_root = Path(__file__).resolve().parent.parent
    targets = [
        project_root / "YuE2-Studio-3.4.0-portable-windows-x64" / "data" / "kaggle_tunnel_url.txt",
        project_root / "ACE-Step-Studio-3.4.0-portable-windows-x64" / "data" / "kaggle_tunnel_url.txt",
        project_root / "MiniMax-Music3-Studio-3.4.0-portable" / "data" / "kaggle_tunnel_url.txt",
        project_root / "data" / "kaggle_tunnel_url.txt",
    ]
    for t in targets:
        if t.is_file():
            try:
                url = t.read_text(encoding="utf-8").strip()
                if verify_tunnel_health(url, timeout=2.0):
                    return url
            except Exception:
                pass
    return None


def get_recent_ntfy_candidates(topic: str) -> list[str]:
    """Fetches recently announced tunnel URLs from ntfy channels (last 2 hours)."""
    candidates: list[str] = []
    channels = [
        f"https://ntfy.sh/{topic}/json?since=2h&poll=1",
        f"https://ntfy.envs.net/{topic}/json?since=2h&poll=1",
    ]
    known_domains = ("trycloudflare.com", "ngrok-free.app", "ngrok.app", "ngrok.io")
    for ch in channels:
        try:
            req = urllib.request.Request(ch, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                lines = [l for l in resp.read().decode("utf-8", errors="ignore").splitlines() if l.strip()]
                for line in reversed(lines):
                    try:
                        d = json.loads(line)
                        msg = d.get("message", "").strip()
                        if any(dom in msg for dom in known_domains) and msg not in candidates:
                            candidates.append(msg)
                    except Exception:
                        continue
        except Exception:
            continue
    return candidates


def wait_for_tunnel_url(
    topic: str,
    timeout_seconds: int = 600,
    api: object | None = None,
    kernel_id: str | None = None,
) -> str:
    """Waits for server to announce active tunnel URL, tracking Kaggle status."""
    cached = get_cached_tunnel_url()
    if cached:
        logger.info("✓ Найден активный сохранённый туннель: %s", cached)
        return cached

    start_time = time.time()
    last_print = 0
    last_status_check = 0
    current_status = "QUEUED / STARTING"

    has_started = False

    while time.time() - start_time < timeout_seconds:
        now = time.time()
        elapsed = int(now - start_time)

        # Periodically check Kaggle worker status
        if api and kernel_id and (now - last_status_check >= 15):
            last_status_check = now
            try:
                st = api.kernels_status(kernel_id)  # type: ignore[attr-defined]
                st_str = str(getattr(st, "status", st))
                if "RUNNING" in st_str or "QUEUED" in st_str:
                    has_started = True
                    current_status = st_str
                elif "ERROR" in st_str:
                    if (has_started or elapsed >= 90):
                        sys.stdout.write("\n")
                        raise RuntimeError(f"Ядро Kaggle завершилось с ошибкой (статус: {st_str}).")
                else:
                    current_status = st_str
            except RuntimeError:
                raise
            except Exception:
                pass

        if elapsed - last_print >= 5:
            last_print = elapsed
            sys.stdout.write(
                f"\r[2/3] ⏳ Ожидание ngrok ({elapsed}с / {timeout_seconds}с) [Kaggle: {current_status}]... "
            )
            sys.stdout.flush()

        candidates = get_recent_ntfy_candidates(topic)
        for cand in candidates[:2]:  # Check at most 2 freshest candidates
            if verify_tunnel_health(cand, timeout=2.0):
                sys.stdout.write("\n")
                return cand

        time.sleep(2.5)

    sys.stdout.write("\n")
    raise TimeoutError(f"Время ожидания туннеля истекло ({timeout_seconds}с).")


def save_tunnel_url(tunnel_url: str) -> None:
    """Saves discovered tunnel URL to data directories for local engines."""
    project_root = Path(__file__).resolve().parent.parent
    targets = [
        project_root / "YuE2-Studio-3.4.0-portable-windows-x64" / "data" / "kaggle_tunnel_url.txt",
        project_root / "ACE-Step-Studio-3.4.0-portable-windows-x64" / "data" / "kaggle_tunnel_url.txt",
        project_root / "data" / "kaggle_tunnel_url.txt",
    ]
    for t in targets:
        try:
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_text(tunnel_url.strip(), encoding="utf-8")
        except Exception:
            pass


def is_process_running(exe_name: str) -> bool:
    """Checks if a process with the given name is currently active."""
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"IMAGENAME eq {exe_name}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return exe_name.lower() in out.lower()
    except Exception:
        return False


def launch_desktop_app(app_choice: str = "auto") -> None:
    """Launches desktop application if not already running."""
    project_root = Path(__file__).resolve().parent.parent
    apps = []

    if app_choice in ("yue2", "auto"):
        apps.append(("YuE2-Studio.exe", project_root / "YuE2-Studio-3.4.0-portable-windows-x64" / "YuE2-Studio.exe"))
    if app_choice in ("acestep",):
        apps.append(("ACE-Step-Studio.exe", project_root / "ACE-Step-Studio-3.4.0-portable-windows-x64" / "ACE-Step-Studio.exe"))
    if app_choice == "both":
        apps.append(("YuE2-Studio.exe", project_root / "YuE2-Studio-3.4.0-portable-windows-x64" / "YuE2-Studio.exe"))
        apps.append(("ACE-Step-Studio.exe", project_root / "ACE-Step-Studio-3.4.0-portable-windows-x64" / "ACE-Step-Studio.exe"))

    for exe_name, exe_path in apps:
        if not exe_path.exists():
            continue
        if is_process_running(exe_name):
            logger.info("✓ Приложение %s уже запущено.", exe_name)
        else:
            logger.info("Запуск настольного приложения: %s", exe_name)
            try:
                if sys.platform == "win32":
                    subprocess.Popen(
                        f'start "" "{exe_name}"',
                        cwd=str(exe_path.parent),
                        shell=True,
                    )
                else:
                    subprocess.Popen([str(exe_path)], cwd=str(exe_path.parent))
            except Exception as exc:
                logger.warning("Не удалось запустить %s: %s", exe_name, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="YuE2 & ACE-Step Studio Cloud Launcher")
    parser.add_argument("--dry-run", action="store_true", help="Validate credentials without launching")
    parser.add_argument(
        "--app",
        choices=["yue2", "acestep", "both", "auto"],
        default="auto",
        help="Target app (default: auto)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("   🎵 YuE2 / ACE-Step Studio — Облачный запуск Kaggle (Tesla T4)")
    print("=" * 65)

    username = ensure_kaggle_credentials()
    topic = compute_ntfy_topic(username, app=args.app)

    if args.dry_run:
        logger.info("Dry-run: пользователь %s, топик: %s", username, topic)
        return

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    if args.app == "acestep":
        kernel_slug = "acestep-audio-generation-experiment"
        kernel_sub = "acestep"
    else:
        kernel_slug = "yue2-music-synthesis-experiment"
        kernel_sub = "yue2"

    kernel_id = f"{username}/{kernel_slug}"

    # 1. Check status on Kaggle
    logger.info("[1/3] Проверка статуса в облаке Kaggle (%s)...", kernel_id)
    kernel_status = "UNKNOWN"
    try:
        st_obj = api.kernels_status(kernel_id)
        kernel_status = str(getattr(st_obj, "status", st_obj))
    except Exception as exc:
        logger.warning("Не удалось получить статус ядра: %s", exc)

    logger.info("Текущий статус ядра: %s", kernel_status)

    if "RUNNING" in kernel_status or "QUEUED" in kernel_status:
        logger.info("✓ Ядро уже активно в Kaggle (%s). Ожидание подключения...", kernel_status)
    else:
        # Prevent concurrent GPU sessions on the same account
        other_slug = "acestep-audio-generation-experiment" if kernel_sub == "yue2" else "yue2-music-synthesis-experiment"
        try:
            other_st = api.kernels_status(f"{username}/{other_slug}")
            other_status_str = str(getattr(other_st, "status", other_st))
            if "RUNNING" in other_status_str or "QUEUED" in other_status_str:
                logger.warning(
                    "⚠️ ВНИМАНИЕ: Другое ядро (%s) уже активно (%s)!",
                    other_slug,
                    other_status_str,
                )
                logger.warning(
                    "Одновременный запуск двух ядер на бесплатном аккаунте может привести к блокировке Kaggle."
                )
        except Exception:
            pass

        kernel_dir = Path(__file__).resolve().parent / kernel_sub
        logger.info("🚀 Запуск нового ядра в Kaggle (%s)...", kernel_sub)
        try:
            resp = api.kernels_push(str(kernel_dir))
            logger.info("✓ Ядро успешно запущено в Kaggle! Версия: %s", getattr(resp, "versionNumber", "OK"))
        except Exception as exc:
            logger.error("Ошибка отправки ядра в Kaggle: %s", exc)
            sys.exit(1)

    # 2. Wait for ngrok tunnel URL
    logger.info("[2/3] Ожидание высокоскоростного туннеля ngrok...")
    try:
        tunnel_url = wait_for_tunnel_url(
            topic,
            timeout_seconds=600,
            api=api,
            kernel_id=kernel_id,
        )
    except (TimeoutError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # 3. Save URL and launch desktop app
    save_tunnel_url(tunnel_url)

    tunnel_type = "ngrok (авторизованный)" if "ngrok" in tunnel_url else "Cloudflare"
    print("\n" + "=" * 65)
    print("🎉 СЕРВЕР KAGGLE ГОТОВ К РАБОТЕ!")
    print(f"⚡ Провайдер: {tunnel_type}")
    print(f"🔗 URL: {tunnel_url}")
    print("=" * 65 + "\n")

    launch_desktop_app(args.app)

    print("[i] Сервер на Kaggle работает в фоне.")
    print("    Для остановки сервера используйте stop-kaggle.bat.")
    print("    Это окно можно закрыть в любой момент.")
    time.sleep(3)


if __name__ == "__main__":
    main()
