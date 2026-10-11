"""YuE2 Studio — Zero-Tunnel Batch Audio Generator (Kaggle API).

100% Ban-Proof ML batch runner:
1. Injects song parameters into the batch template.
2. Pushes kernel to Kaggle Tesla T4 via official Kaggle API.
3. Monitors execution progress until completion.
4. Downloads the resulting output.mp3 directly to local disk.
Zero tunnels, zero daemons, zero ToS violations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("zero-tunnel-gen")

DEFAULT_USERNAME = "jungterdai"
KERNEL_SLUG = "yue2-music-generator"


def run_batch_generation(
    title: str,
    style: str,
    lyrics: str,
    duration: int = 120,
    steps: int = 32,
    seed: int = 42,
    output_dir: Path | None = None,
) -> Path:
    """Submits generation job to Kaggle and retrieves output.mp3."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    username = DEFAULT_USERNAME
    kernel_id = f"{username}/{KERNEL_SLUG}"

    project_root = Path(__file__).resolve().parent
    batch_dir = project_root / "kaggle" / "batch"
    main_py = batch_dir / "main.py"

    if not main_py.exists():
        raise FileNotFoundError(f"Missing batch template: {main_py}")

    logger.info("Configuring task specification...")
    task_spec = {
        "title": title,
        "style": style,
        "lyrics": lyrics,
        "duration": duration,
        "steps": steps,
        "seed": seed,
        "abc": None,
        "username": username,
    }

    # Inject task specification into template
    content = main_py.read_text(encoding="utf-8")
    marker_start = "# === TASK SPECIFICATION"
    marker_end = "# ===================================================="
    
    new_spec_block = (
        f"{marker_start} (injected per generation) ===\n"
        f"TASK_SPEC = {json.dumps(task_spec, indent=4, ensure_ascii=False)}\n"
        f"{marker_end}"
    )

    if marker_start in content and marker_end in content:
        before = content.split(marker_start)[0]
        after = content.split(marker_end)[1]
        updated_content = before + new_spec_block + after
        main_py.write_text(updated_content, encoding="utf-8")

    logger.info("Submitting job to Kaggle GPU (%s)...", kernel_id)
    resp = api.kernels_push(str(batch_dir))
    ver = getattr(resp, "versionNumber", "1")
    logger.info("✓ Задача принята Kaggle (версия: %s). GPU начинает синтез...", ver)

    start_time = time.time()
    last_print = 0
    while True:
        elapsed = int(time.time() - start_time)
        st_obj = api.kernels_status(kernel_id)
        status = str(getattr(st_obj, "status", st_obj)).upper()

        if elapsed - last_print >= 5:
            last_print = elapsed
            sys.stdout.write(f"\r⏳ Синтез в Kaggle на 2x Tesla T4 ({elapsed}с) [Статус: {status}]... ")
            sys.stdout.flush()

        if status == "COMPLETE":
            sys.stdout.write("\n")
            logger.info("🎉 Синтез успешно завершён!")
            break
        elif "ERROR" in status or "FAILED" in status:
            sys.stdout.write("\n")
            raise RuntimeError(f"Задача на Kaggle завершилась с ошибкой: {status}")

        time.sleep(4)

    # Download output
    dest_dir = output_dir or (project_root / "generated_tracks")
    dest_dir.mkdir(parents=True, exist_ok=True)
    temp_download = Path(tempfile.mkdtemp())

    logger.info("Загрузка сгенерированного трека через Kaggle API...")
    api.kernels_output(kernel_id, str(temp_download))

    found_mp3: Path | None = None
    for f in temp_download.rglob("*.mp3"):
        if f.is_file() and f.stat().st_size > 1024:
            found_mp3 = f
            break

    if not found_mp3:
        # Check if output is named differently
        for f in temp_download.iterdir():
            if f.is_file() and f.stat().st_size > 1024:
                found_mp3 = f
                break

    if not found_mp3:
        raise FileNotFoundError(f"MP3 файл не найден в выводе Kaggle: {list(temp_download.iterdir())}")

    safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).strip() or "track"
    final_path = dest_dir / f"{safe_title}.mp3"
    shutil.copy2(found_mp3, final_path)
    shutil.rmtree(temp_download, ignore_errors=True)

    logger.info("✓ Трек успешно сохранён: %s (%d КБ)", final_path, final_path.stat().st_size // 1024)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="YuE2 Zero-Tunnel Batch Generator")
    parser.add_argument("--title", default="YuE2 Synthwave Track", help="Название трека")
    parser.add_argument(
        "--style",
        default="Synthwave, 80s analog synthesizer, punchy drums, driving bass",
        help="Стиль и инструменты",
    )
    parser.add_argument(
        "--lyrics",
        default="[Intro]\n[Verse 1]\nОгни ночного города горят вдали\n[Chorus]\nМузыка звучит сквозь века летит",
        help="Текст песни и структура",
    )
    parser.add_argument("--duration", type=int, default=120, help="Длительность в секундах")
    parser.add_argument("--steps", type=int, default=32, help="Шаги Flow-Matching (default: 32)")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно")

    args = parser.parse_args()
    print("=" * 65)
    print("🎵 YuE2 Studio — Генерация без туннелей (Kaggle Batch API)")
    print("=" * 65)

    try:
        mp3_path = run_batch_generation(
            title=args.title,
            style=args.style,
            lyrics=args.lyrics,
            duration=args.duration,
            steps=args.steps,
            seed=args.seed,
        )
        print("\n" + "=" * 65)
        print(f"🎉 ГОТОВО! Файл сохранён: {mp3_path}")
        print("=" * 65)
        # Open in default player
        if sys.platform == "win32":
            os.startfile(str(mp3_path))
    except Exception as exc:
        logger.error("Ошибка генерации: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
