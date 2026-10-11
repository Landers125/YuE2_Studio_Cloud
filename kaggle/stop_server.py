"""Stops local bridge proxy and signals remote Kaggle instance to shut down."""

import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("stop-kaggle")

def main() -> None:
    print("=" * 60)
    print("🛑 Остановка серверов YuE2 Studio Kaggle...")
    print("=" * 60)

    # 1. Попытка остановки через локальный мост или прямой URL туннеля
    stopped_remote = False
    urls_to_try = ["http://127.0.0.1:18087/shutdown"]

    project_root = Path(__file__).resolve().parent.parent
    tunnel_files = [
        project_root / "data" / "kaggle_tunnel_url.txt",
        project_root / "YuE2-Studio-3.4.0-portable-windows-x64" / "data" / "kaggle_tunnel_url.txt",
        project_root / "ACE-Step-Studio-3.4.0-portable-windows-x64" / "data" / "kaggle_tunnel_url.txt",
    ]
    for tf in tunnel_files:
        if tf.is_file():
            try:
                u = tf.read_text(encoding="utf-8").strip().rstrip("/")
                if u and f"{u}/shutdown" not in urls_to_try:
                    urls_to_try.append(f"{u}/shutdown")
            except Exception:
                pass

    for url in urls_to_try:
        try:
            req = urllib.request.Request(url, headers={"ngrok-skip-browser-warning": "true"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                print(f"✓ Сигнал остановки успешно отправлен удаленному серверу Kaggle ({url})!")
                stopped_remote = True
                break
        except Exception:
            pass

    # 2. Завершение локальных процессов launcher
    try:
        res = subprocess.run(
            ["taskkill", "/F", "/FI", "WINDOWTITLE eq *YuE2 Studio - Kaggle 2x T4*"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

    # 3. Проверка статуса в облаке Kaggle
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        username = "jungterdai"
        kernels_to_check = [
            ("YuE2", f"{username}/yue2-music-synthesis-experiment"),
            ("ACE-Step", f"{username}/acestep-audio-generation-experiment"),
        ]
        for name, kid in kernels_to_check:
            try:
                status_res = api.kernels_status(kid)
                status = str(getattr(status_res, "status", status_res))
                print(f"📡 Статус ядра {name} в Kaggle: {status}")
            except Exception:
                pass
    except Exception as exc:
        if stopped_remote:
            print("✓ Сервер успешно получил сигнал остановки.")
        else:
            print(f"ℹ️ {exc}")

    print("=" * 60)

if __name__ == "__main__":
    main()
