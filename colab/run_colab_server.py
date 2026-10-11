"""YuE2 Studio external engine runner for Google Colab (Tesla T4 GPU).

Provides automated environment verification, yue2.cpp compilation,
model weight downloading via aria2c/HuggingFace, Cloudflare/ngrok tunneling,
and supervisor for the yue-server inference process.
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger: logging.Logger = logging.getLogger("yue2-colab")

HF_REPO_YUE_GGUF: Final[str] = "Serveurperso/YuE2-GGUF"
HF_REV_YUE_GGUF: Final[str] = "64b030e3deb6e8150d2b7c0db641ef5a17eca8a3"
HF_REPO_COMPANION: Final[str] = (
    "Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4"
)
HF_REV_COMPANION: Final[str] = "e2e63d859f3af879baf1b4d4e9f22d1eeda6fde5"

DEFAULT_PORT: Final[int] = 18087
CLOUDFLARED_URL: Final[str] = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
)


@dataclass(frozen=True)
class ModelFileConfig:
    """Download specification for a single model weight file."""

    filename: str
    url: str
    size_mb: int
    role: str


MODEL_CATALOG: Final[dict[str, list[ModelFileConfig]]] = {
    "q8_0": [
        ModelFileConfig(
            filename="YuE2-3B-Q8_0.gguf",
            url=f"https://huggingface.co/{HF_REPO_YUE_GGUF}/resolve/{HF_REV_YUE_GGUF}/YuE2-3B-Q8_0.gguf",
            size_mb=3550,
            role="backbone",
        ),
        ModelFileConfig(
            filename="YuE2-Vae-F32.gguf",
            url=f"https://huggingface.co/{HF_REPO_YUE_GGUF}/resolve/{HF_REV_YUE_GGUF}/YuE2-Vae-F32.gguf",
            size_mb=506,
            role="vae",
        ),
        ModelFileConfig(
            filename="nar_lora_joint_v9.safetensors",
            url=f"https://huggingface.co/{HF_REPO_COMPANION}/resolve/{HF_REV_COMPANION}/nar_lora_joint_v9.safetensors",
            size_mb=134,
            role="companion",
        ),
        ModelFileConfig(
            filename="SheetSage2-Q8_0.gguf",
            url=f"https://huggingface.co/{HF_REPO_YUE_GGUF}/resolve/{HF_REV_YUE_GGUF}/SheetSage2-Q8_0.gguf",
            size_mb=913,
            role="transcriber",
        ),
    ],
    "bf16": [
        ModelFileConfig(
            filename="YuE2-3B-BF16.gguf",
            url=f"https://huggingface.co/{HF_REPO_YUE_GGUF}/resolve/{HF_REV_YUE_GGUF}/YuE2-3B-BF16.gguf",
            size_mb=6670,
            role="backbone",
        ),
        ModelFileConfig(
            filename="YuE2-Vae-F32.gguf",
            url=f"https://huggingface.co/{HF_REPO_YUE_GGUF}/resolve/{HF_REV_YUE_GGUF}/YuE2-Vae-F32.gguf",
            size_mb=506,
            role="vae",
        ),
        ModelFileConfig(
            filename="nar_lora_joint_v9.safetensors",
            url=f"https://huggingface.co/{HF_REPO_COMPANION}/resolve/{HF_REV_COMPANION}/nar_lora_joint_v9.safetensors",
            size_mb=134,
            role="companion",
        ),
        ModelFileConfig(
            filename="SheetSage2-F32.gguf",
            url=f"https://huggingface.co/{HF_REPO_YUE_GGUF}/resolve/{HF_REV_YUE_GGUF}/SheetSage2-F32.gguf",
            size_mb=2520,
            role="transcriber",
        ),
    ],
    "q6_k": [
        ModelFileConfig(
            filename="YuE2-3B-Q6_K.gguf",
            url=f"https://huggingface.co/{HF_REPO_YUE_GGUF}/resolve/{HF_REV_YUE_GGUF}/YuE2-3B-Q6_K.gguf",
            size_mb=2740,
            role="backbone",
        ),
        ModelFileConfig(
            filename="YuE2-Vae-F32.gguf",
            url=f"https://huggingface.co/{HF_REPO_YUE_GGUF}/resolve/{HF_REV_YUE_GGUF}/YuE2-Vae-F32.gguf",
            size_mb=506,
            role="vae",
        ),
        ModelFileConfig(
            filename="nar_lora_joint_v9.safetensors",
            url=f"https://huggingface.co/{HF_REPO_COMPANION}/resolve/{HF_REV_COMPANION}/nar_lora_joint_v9.safetensors",
            size_mb=134,
            role="companion",
        ),
        ModelFileConfig(
            filename="SheetSage2-Q6_K.gguf",
            url=f"https://huggingface.co/{HF_REPO_YUE_GGUF}/resolve/{HF_REV_YUE_GGUF}/SheetSage2-Q6_K.gguf",
            size_mb=776,
            role="transcriber",
        ),
    ],
    "q5_k_m": [
        ModelFileConfig(
            filename="YuE2-3B-Q5_K_M.gguf",
            url=f"https://huggingface.co/{HF_REPO_YUE_GGUF}/resolve/{HF_REV_YUE_GGUF}/YuE2-3B-Q5_K_M.gguf",
            size_mb=2440,
            role="backbone",
        ),
        ModelFileConfig(
            filename="YuE2-Vae-F32.gguf",
            url=f"https://huggingface.co/{HF_REPO_YUE_GGUF}/resolve/{HF_REV_YUE_GGUF}/YuE2-Vae-F32.gguf",
            size_mb=506,
            role="vae",
        ),
        ModelFileConfig(
            filename="nar_lora_joint_v9.safetensors",
            url=f"https://huggingface.co/{HF_REPO_COMPANION}/resolve/{HF_REV_COMPANION}/nar_lora_joint_v9.safetensors",
            size_mb=134,
            role="companion",
        ),
        ModelFileConfig(
            filename="SheetSage2-Q5_K_M.gguf",
            url=f"https://huggingface.co/{HF_REPO_YUE_GGUF}/resolve/{HF_REV_YUE_GGUF}/SheetSage2-Q5_K_M.gguf",
            size_mb=703,
            role="transcriber",
        ),
    ],
}


def check_gpu() -> bool:
    """Checks NVIDIA GPU availability and CUDA compute capability."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
        logger.info("Found GPU: %s", output)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.warning("No NVIDIA GPU detected via nvidia-smi!")
        return False


def ensure_system_packages() -> None:
    """Installs aria2, cmake, ninja-build if missing in Colab/Linux environment."""
    missing: list[str] = []
    if shutil.which("aria2c") is None:
        missing.append("aria2")
    if shutil.which("cmake") is None:
        missing.append("cmake")
    if shutil.which("ninja") is None:
        missing.append("ninja-build")
    if missing and shutil.which("apt-get"):
        logger.info("Installing missing system packages: %s", ", ".join(missing))
        try:
            subprocess.check_call(["apt-get", "update", "-qq"])
            subprocess.check_call(
                ["apt-get", "install", "-y", "-qq", "--no-install-recommends"] + missing
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            logger.warning("Package installation notice: %s", exc)

    if shutil.which("muscriptor") is None:
        logger.info("Installing muscriptor for Tesla T4 GPU Audio-to-MIDI...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "muscriptor"]
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            logger.warning("Muscriptor pip notice: %s", exc)


def build_yue_engine(work_dir: Path) -> Path:
    """Restores yue-server and shared libs from Google Drive or compiles yue2.cpp for Tesla T4."""
    local_target = work_dir / "yue-server"
    drive_base = Path("/content/drive/MyDrive")
    drive_dir = drive_base / "YuE2"
    drive_target = drive_dir / "yue-server"

    # Check if we have both yue-server AND its companion shared libraries (*.so*)
    drive_libs = list(drive_dir.glob("lib*.so*")) if drive_dir.exists() else []

    if local_target.exists():
        local_libs = list(work_dir.glob("lib*.so*"))
        if local_libs:
            local_target.chmod(0o755)
            logger.info("Found cached yue-server with libraries at %s", local_target)
            return local_target

    if drive_target.exists() and drive_libs:
        logger.info(
            "Restoring compiled yue-server and shared libraries from Google Drive (%s)...",
            drive_dir,
        )
        try:
            shutil.copy2(drive_target, local_target)
            local_target.chmod(0o755)
            for lib in drive_libs:
                shutil.copy2(lib, work_dir / lib.name)
            logger.info(
                "Restored yue-server and %d libraries ready at %s",
                len(drive_libs),
                local_target,
            )
            return local_target
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not restore from Drive: %s, will compile", exc)
    elif drive_target.exists() and not drive_libs:
        logger.info(
            "Found yue-server on Drive, but missing libggml.so companion libraries. "
            "Building yue2.cpp once to produce and save full binary package..."
        )

    repo_dir = work_dir / "yue2.cpp"
    if not repo_dir.exists():
        logger.info("Cloning yue2.cpp repository...")
        try:
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--recurse-submodules",
                    "https://github.com/timoncool/yue2.cpp.git",
                    str(repo_dir),
                ]
            )
        except subprocess.CalledProcessError:
            logger.info("Falling back to ServeurpersoCom/yue2.cpp...")
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--recurse-submodules",
                    "https://github.com/ServeurpersoCom/yue2.cpp.git",
                    str(repo_dir),
                ]
            )

    build_dir = repo_dir / "build"
    executable = build_dir / "yue-server"
    if not executable.exists():
        executable = build_dir / "bin" / "yue-server"

    if not executable.exists():
        logger.info("Configuring CMake for Tesla T4 (sm_75) CUDA build...")
        subprocess.check_call(
            [
                "cmake",
                "-S",
                str(repo_dir),
                "-B",
                str(build_dir),
                "-DGGML_CUDA=ON",
                "-DCMAKE_CUDA_ARCHITECTURES=75",
                "-DCMAKE_BUILD_TYPE=Release",
            ]
        )

        cpu_count = os.cpu_count() or 2
        logger.info("Compiling yue-server using %d threads...", cpu_count)
        subprocess.check_call(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "-j",
                str(cpu_count),
            ]
        )

        target_exe = build_dir / "yue-server"
        if not target_exe.exists():
            target_exe = build_dir / "bin" / "yue-server"
        if not target_exe.exists():
            raise RuntimeError(
                f"Build succeeded but yue-server binary not found in {build_dir}"
            )
    else:
        target_exe = executable

    shutil.copy2(target_exe, local_target)
    local_target.chmod(0o755)
    logger.info("Successfully built yue-server: %s", local_target)

    # Collect and copy all shared libraries from build directory
    so_files: list[Path] = []
    for f in build_dir.rglob("*.so*"):
        if f.is_file() and not f.is_symlink():
            so_files.append(f)
            try:
                shutil.copy2(f, work_dir / f.name)
            except Exception:
                pass
        elif f.is_symlink():
            so_files.append(f)
            try:
                target = f.resolve()
                shutil.copy2(target, work_dir / f.name)
            except Exception:
                pass

    logger.info("Copied %d shared libraries to %s", len(so_files), work_dir)

    # Persist executable AND all *.so* libraries to Google Drive
    if drive_base.exists():
        drive_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(local_target, drive_target)
            for f in so_files:
                shutil.copy2(work_dir / f.name, drive_dir / f.name)
            logger.info(
                "Saved compiled yue-server and %d libraries to Google Drive (%s)!",
                len(so_files),
                drive_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not backup to Drive: %s", exc)

    return local_target


def download_models(
    models_dir: Path, profile: str, include_transcriber: bool = True
) -> dict[str, Path]:
    """Downloads required GGUF weights into models_dir or reuses Google Drive cache."""
    models_dir.mkdir(parents=True, exist_ok=True)
    drive_dir = Path("/content/drive/MyDrive/YuE2")
    drive_models_dir = drive_dir / "models"
    configs = MODEL_CATALOG.get(profile, MODEL_CATALOG["q8_0"])
    has_aria2c = shutil.which("aria2c") is not None
    resolved: dict[str, Path] = {}

    for cfg in configs:
        if cfg.role == "transcriber" and not include_transcriber:
            continue

        min_bytes = int(cfg.size_mb * 1024 * 1024 * 0.9)

        # 1. Check Google Drive cache
        drive_path1 = drive_dir / cfg.filename
        drive_path2 = drive_models_dir / cfg.filename
        if drive_path1.exists() and drive_path1.stat().st_size > min_bytes:
            logger.info("Found '%s' on Google Drive (%s)", cfg.filename, drive_path1)
            resolved[cfg.role] = drive_path1
            continue
        if drive_path2.exists() and drive_path2.stat().st_size > min_bytes:
            logger.info("Found '%s' on Google Drive (%s)", cfg.filename, drive_path2)
            resolved[cfg.role] = drive_path2
            continue

        # 2. Check local models dir
        dest = models_dir / cfg.filename
        resolved[cfg.role] = dest
        if dest.exists() and dest.stat().st_size > min_bytes:
            logger.info("Model '%s' already exists (%d MB).", cfg.filename, cfg.size_mb)
            continue

        # 3. Target directory: save to Google Drive if available
        save_dir = drive_models_dir if drive_dir.exists() else models_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        target_file = save_dir / cfg.filename
        resolved[cfg.role] = target_file

        logger.info(
            "Downloading %s (~%d MB) into %s...", cfg.filename, cfg.size_mb, save_dir
        )
        if has_aria2c:
            subprocess.check_call(
                [
                    "aria2c",
                    "-c",
                    "-x",
                    "16",
                    "-s",
                    "16",
                    "-k",
                    "1M",
                    "--file-allocation=none",
                    "--dir",
                    str(save_dir),
                    "--out",
                    cfg.filename,
                    cfg.url,
                ]
            )
        else:
            urllib.request.urlretrieve(cfg.url, str(target_file))

    return resolved


def setup_cloudflare_tunnel(
    port: int, work_dir: Path
) -> tuple[subprocess.Popen[str], str]:
    """Downloads cloudflared and starts an ad-hoc tunnel to localhost:port."""
    cf_bin = work_dir / "cloudflared"
    if not cf_bin.exists() and shutil.which("cloudflared") is None:
        logger.info("Downloading cloudflared tunnel client...")
        urllib.request.urlretrieve(CLOUDFLARED_URL, str(cf_bin))
        cf_bin.chmod(0o755)

    binary = str(cf_bin) if cf_bin.exists() else "cloudflared"
    cmd = [binary, "tunnel", "--url", f"http://127.0.0.1:{port}"]
    logger.info("Launching Cloudflare Tunnel: %s", " ".join(cmd))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tunnel_url: str | None = None
    start_time = time.time()
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    while time.time() - start_time < 30:
        line = process.stdout.readline() if process.stdout else ""
        if not line:
            time.sleep(0.1)
            continue
        match = url_pattern.search(line)
        if match:
            tunnel_url = match.group(0)
            break

    if not tunnel_url:
        process.kill()
        raise RuntimeError("Failed to obtain Cloudflare tunnel URL within 30 seconds.")

    return process, tunnel_url


def setup_ngrok_tunnel(
    port: int, token: str | None = None
) -> tuple[subprocess.Popen[str] | None, str]:
    """Sets up ngrok tunnel using pyngrok or ngrok binary."""
    try:
        from pyngrok import ngrok  # type: ignore[import-untyped]
    except ImportError:
        logger.info("Installing pyngrok for ngrok tunneling...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "pyngrok"]
        )
        from pyngrok import ngrok  # type: ignore[import-untyped]

    if token:
        ngrok.set_auth_token(token)

    try:
        ngrok.kill()
    except Exception:  # noqa: BLE001, S110
        pass

    tunnel = ngrok.connect(port, "http")
    public_url = str(tunnel.public_url)
    if public_url.startswith("http://"):
        public_url = "https://" + public_url[7:]
    logger.info("ngrok tunnel active: %s", public_url)
    return None, public_url


def launch_engine(
    executable: Path,
    models: dict[str, Path],
    adapters_dir: Path | None = None,
    port: int = DEFAULT_PORT,
    keep_loaded: bool = True,
) -> subprocess.Popen[str]:
    """Runs the yue-server process with the configured models and options."""
    cmd = [
        str(executable),
        "--model",
        str(models["backbone"]),
        "--vae",
        str(models["vae"]),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    if "companion" in models and models["companion"].exists():
        cmd.extend(["--companion", str(models["companion"])])
    if "transcriber" in models and models["transcriber"].exists():
        cmd.extend(["--transcriber", str(models["transcriber"])])
    if adapters_dir and adapters_dir.exists():
        has_adapters = any(p.is_dir() for p in adapters_dir.iterdir())
        if has_adapters:
            cmd.extend(["--adapters", str(adapters_dir)])
            logger.info("Configured LoRA adapters directory: %s", adapters_dir)

    logger.info("Executing engine: %s", " ".join(cmd))
    env = os.environ.copy()
    env["GGML_BACKEND"] = "CUDA0"
    work_dir = executable.parent
    drive_dir = Path("/content/drive/MyDrive/YuE2")
    build_dir = work_dir / "yue2.cpp" / "build"
    env["LD_LIBRARY_PATH"] = (
        f"{work_dir}:{drive_dir}:{build_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    )

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    return process


def print_banner(tunnel_url: str) -> None:
    """Prints user instructions with clickable banner and HTML card in Colab."""
    line = "=" * 70
    msg = (
        f"\n{line}\n"
        f"🚀 YuE2 STUDIO CLOUD ENGINE (TESLA T4) IS READY!\n"
        f"{line}\n"
        f"🔗 ССЫЛКА ДЛЯ connect-colab.bat:\n"
        f"   >>> {tunnel_url} <<<\n"
        f"{line}\n"
        f"📋 Инструкция для подключения:\n"
        f"  1. Скопируй ссылку: {tunnel_url}\n"
        f"  2. Запусти на ПК файл connect-colab.bat и вставь эту ссылку\n"
        f"     (или вставь в YuE2 Studio: Настройки -> Движок -> Внешний сервер)\n"
        f"  3. Нажми 'Проверить' -> 'Сохранить и применить'\n"
        f"  4. Генерация и кнопка 'Сделать MIDI' теперь работают на Tesla T4!\n"
        f"{line}\n"
    )
    print(msg, flush=True)
    sys.stdout.flush()

    try:
        from IPython.display import HTML, display  # type: ignore[import-untyped]

        html_banner = f"""
        <div style="background: linear-gradient(135deg, #181825, #241835); border: 2px solid #ec4899; border-radius: 12px; padding: 20px; color: white; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 15px 0; max-width: 680px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <span style="font-size: 24px;">🚀</span>
                <h3 style="margin: 0; color: #f472b6; font-size: 18px;">YuE2 Studio Cloud Engine (Tesla T4) Готов!</h3>
            </div>
            <p style="font-size: 14px; margin: 0 0 8px 0; color: #cbd5e1;">🔗 <b>Ссылка для <code style="background: rgba(255,255,255,0.15); padding: 2px 6px; border-radius: 4px; color: #f43f5e;">connect-colab.bat</code>:</b></p>
            <div style="background: #0f0f17; border: 1px solid #38bdf8; border-radius: 8px; padding: 12px 16px; font-size: 18px; font-family: monospace; font-weight: bold; color: #38bdf8; word-break: break-all; user-select: all; cursor: pointer;" title="Кликните чтобы выделить">
                {tunnel_url}
            </div>
            <ol style="margin: 14px 0 0 0; padding-left: 20px; line-height: 1.6; font-size: 13px; color: #94a3b8;">
                <li>Скопируйте ссылку выше (кликните по ней для выделения).</li>
                <li>Запустите на компьютере <b>connect-colab.bat</b> и вставьте её.</li>
                <li>В YuE2 Studio создавайте треки и делайте MIDI через Tesla T4!</li>
            </ol>
        </div>
        """
        display(HTML(html_banner))
    except Exception:
        pass
    logger.info("Tunnel ready at %s", tunnel_url)


def ensure_muscriptor_model(work_dir: Path, size: str = "large") -> Path:
    """Ensures MuScriptor weights (small/medium/large) are downloaded on Google Drive or work_dir."""
    size_norm = size.lower().strip()
    if "small" in size_norm or "103" in size_norm:
        key = "small"
    elif "medium" in size_norm or "307" in size_norm:
        key = "medium"
    else:
        key = "large"

    drive_dir = Path("/content/drive/MyDrive/YuE2")
    base_dir = drive_dir if drive_dir.exists() else work_dir
    target_dir = base_dir / "models" / f"muscriptor-{key}"
    target_dir.mkdir(parents=True, exist_ok=True)
    safetensors = target_dir / "model.safetensors"
    config_json = target_dir / "config.json"

    specs = {
        "large": (
            "https://huggingface.co/cocktailpeanut/muscriptor-large/resolve/87f4bf981f56f90fb5043153b3f54af3c3053da9/model.safetensors",
            "https://huggingface.co/cocktailpeanut/muscriptor-large/resolve/87f4bf981f56f90fb5043153b3f54af3c3053da9/config.json",
            5_000_000_000,
        ),
        "medium": (
            "https://huggingface.co/cocktailpeanut/muscriptor-medium/resolve/27246ba68bd4d8f98bdec10a6edf8d7cf42a8826/model.safetensors",
            "https://huggingface.co/cocktailpeanut/muscriptor-medium/resolve/27246ba68bd4d8f98bdec10a6edf8d7cf42a8826/config.json",
            1_000_000_000,
        ),
        "small": (
            "https://huggingface.co/cocktailpeanut/muscriptor-small/resolve/31a8f75d6a8b5383fd71ad1371dc1620389ab722/model.safetensors",
            "https://huggingface.co/cocktailpeanut/muscriptor-small/resolve/31a8f75d6a8b5383fd71ad1371dc1620389ab722/config.json",
            300_000_000,
        ),
    }

    url_st, url_cfg, min_bytes = specs[key]

    if not safetensors.exists() or safetensors.stat().st_size < min_bytes:
        logger.info(
            "Downloading MuScriptor %s weights into %s...", key, target_dir
        )
        has_aria2 = shutil.which("aria2c") is not None
        if has_aria2:
            subprocess.check_call(
                [
                    "aria2c",
                    "-c",
                    "-x",
                    "16",
                    "-s",
                    "16",
                    "-k",
                    "1M",
                    "--file-allocation=none",
                    "--dir",
                    str(target_dir),
                    "--out",
                    "model.safetensors",
                    url_st,
                ]
            )
            subprocess.check_call(
                [
                    "aria2c",
                    "-c",
                    "--dir",
                    str(target_dir),
                    "--out",
                    "config.json",
                    url_cfg,
                ]
            )
        else:
            urllib.request.urlretrieve(url_st, str(safetensors))
            urllib.request.urlretrieve(url_cfg, str(config_json))
        logger.info("MuScriptor %s weights ready at %s", key, safetensors)

    return safetensors


class UnifiedServerHandler(http.server.BaseHTTPRequestHandler):
    """Unified HTTP handler: proxies generation to yue-server, handles /transcribe via MuScriptor."""

    backend_port: int = 18088
    muscriptor_model: Path = Path(
        "/content/drive/MyDrive/YuE2/models/muscriptor-large/model.safetensors"
    )

    def do_GET(self) -> None:
        self._proxy("GET")

    def do_POST(self) -> None:
        if self.path.startswith("/transcribe"):
            self._handle_transcribe()
            return
        self._proxy("POST")

    def do_HEAD(self) -> None:
        self._proxy("HEAD")

    def _proxy(self, method: str) -> None:
        target = f"http://127.0.0.1:{self.backend_port}{self.path}"
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        req_headers: dict[str, str] = {}
        for header, value in self.headers.items():
            if header.lower() not in ("host", "content-length"):
                req_headers[header] = value

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
            self.send_response(exc.code)
            self.end_headers()
            self.wfile.write(exc.read())
        except Exception as exc:  # noqa: BLE001
            logger.error("Internal proxy error: %s", exc)
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(exc).encode())

    def _handle_transcribe(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Empty audio payload")
                return

            model_size = self.headers.get("X-Model-Size", "large").strip().lower()
            audio_data = self.rfile.read(content_length)
            tmp_in = Path("/tmp/input_transcribe.mp3")
            tmp_out = Path("/tmp/output_transcribe.mid")
            tmp_in.write_bytes(audio_data)

            if tmp_out.exists():
                tmp_out.unlink()

            if shutil.which("muscriptor") is None:
                logger.info("Installing muscriptor package...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", "muscriptor"]
                )

            model_path = ensure_muscriptor_model(Path("/content"), size=model_size)

            logger.info(
                "Transcribing audio (%d bytes, model=%s) with MuScriptor on Tesla T4...",
                len(audio_data),
                model_size,
            )
            cmd = [
                "muscriptor",
                "transcribe",
                str(tmp_in),
                "-o",
                str(tmp_out),
                "--model",
                str(model_path),
                "--device",
                "cuda",
                "--dtype",
                "float16",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                err_text = (res.stderr or res.stdout or "").strip()
                logger.warning(
                    "MuScriptor CUDA attempt failed (code %d): %s",
                    res.returncode,
                    err_text,
                )
                logger.info("Retrying MuScriptor on CPU...")
                cmd_cpu = [
                    "muscriptor",
                    "transcribe",
                    str(tmp_in),
                    "-o",
                    str(tmp_out),
                    "--model",
                    str(model_path),
                    "--device",
                    "cpu",
                    "--dtype",
                    "float32",
                ]
                res_cpu = subprocess.run(cmd_cpu, capture_output=True, text=True)
                if res_cpu.returncode != 0:
                    err_cpu = (res_cpu.stderr or res_cpu.stdout or "").strip()
                    logger.error("MuScriptor CPU attempt also failed: %s", err_cpu)
                    raise RuntimeError(f"MuScriptor failed: {err_text or err_cpu}")

            if not tmp_out.exists():
                raise RuntimeError("muscriptor completed but output .mid file was not created")

            midi_bytes = tmp_out.read_bytes()
            logger.info("Transcription completed! MIDI size: %d bytes", len(midi_bytes))

            self.send_response(200)
            self.send_header("Content-Type", "audio/midi")
            self.send_header("Content-Length", str(len(midi_bytes)))
            self.end_headers()
            self.wfile.write(midi_bytes)

            if tmp_in.exists():
                tmp_in.unlink()
            if tmp_out.exists():
                tmp_out.unlink()
        except Exception as exc:  # noqa: BLE001
            logger.error("Transcribe error: %s", exc)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Transcribe error: {exc}".encode())

    def log_message(self, format: str, *args: object) -> None:
        if args and str(args[0]).startswith("GET /health"):
            return
        logger.info("%s - %s", self.address_string(), format % args)


class ReusableServer(http.server.ThreadingHTTPServer):
    """Threading HTTPServer with socket SO_REUSEADDR enabled."""

    allow_reuse_address = True


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run YuE2-Studio backend on Google Colab (Tesla T4)"
    )
    parser.add_argument(
        "--model",
        "--quant",
        dest="model",
        type=str,
        default="q8_0",
        help="Model quantisation profile (default: q8_0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Engine port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--tunnel",
        choices=["cloudflare", "ngrok", "none"],
        default="cloudflare",
        help="Tunneling method to expose engine URL (default: cloudflare)",
    )
    parser.add_argument(
        "--ngrok-token",
        type=str,
        default="",
        help="ngrok authtoken if using --tunnel ngrok",
    )
    parser.add_argument(
        "--keep-loaded",
        dest="keep_loaded",
        action="store_true",
        default=True,
        help="Keep model in GPU memory across requests (default: True)",
    )
    parser.add_argument(
        "--no-keep-loaded",
        dest="keep_loaded",
        action="store_false",
        help="Unload model from GPU memory between requests",
    )
    parser.add_argument(
        "--no-transcriber",
        action="store_true",
        help="Skip downloading SheetSage2 cover transcriber",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/content"),
        help="Working directory in Google Colab (default: /content)",
    )

    args = parser.parse_args()
    run_server(
        model=args.model,
        port=args.port,
        tunnel=args.tunnel,
        ngrok_token=args.ngrok_token,
        keep_loaded=args.keep_loaded,
        no_transcriber=args.no_transcriber,
        work_dir=args.work_dir,
    )


def run_server(
    model: str = "q8_0",
    port: int = DEFAULT_PORT,
    tunnel: str = "cloudflare",
    ngrok_token: str = "",
    keep_loaded: bool = True,
    no_transcriber: bool = False,
    work_dir: Path = Path("/content"),
) -> None:
    """Initializes models, compiler, HTTP supervisor and tunnel."""
    models_dir = work_dir / "models"
    work_dir.mkdir(parents=True, exist_ok=True)

    backend_port = port + 1

    # Cleanup any stale processes from previous Colab runs
    try:
        subprocess.call(
            ["fuser", "-k", f"{port}/tcp", f"{backend_port}/tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.call(
            ["killall", "-9", "cloudflared"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
    except Exception:  # noqa: BLE001, S110
        pass

    ensure_system_packages()
    check_gpu()
    engine_exe = build_yue_engine(work_dir)
    models = download_models(
        models_dir,
        model.lower(),
        include_transcriber=not no_transcriber,
    )
    drive_dir = Path("/content/drive/MyDrive/YuE2")
    drive_adapters = drive_dir / "adapters"
    local_adapters = work_dir / "adapters"
    adapters_dir = drive_adapters if drive_dir.exists() else local_adapters
    adapters_dir.mkdir(parents=True, exist_ok=True)

    # Launch yue-server on internal port (port + 1)
    engine_proc = launch_engine(
        engine_exe,
        models,
        adapters_dir=adapters_dir,
        port=backend_port,
        keep_loaded=keep_loaded,
    )

    # Start Unified HTTP Server on port (handles /transcribe + proxies yue-server)
    UnifiedServerHandler.backend_port = backend_port
    httpd = ReusableServer(("0.0.0.0", port), UnifiedServerHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    tunnel_proc: subprocess.Popen[str] | None = None
    tunnel_url = f"http://127.0.0.1:{port}"

    if tunnel == "cloudflare":
        tunnel_proc, tunnel_url = setup_cloudflare_tunnel(port, work_dir)
    elif tunnel == "ngrok":
        tunnel_proc, tunnel_url = setup_ngrok_tunnel(port, ngrok_token or None)

    print_banner(tunnel_url)

    banner_shown_after_listen = False
    try:
        while True:
            if engine_proc.stdout:
                line = engine_proc.stdout.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    if not banner_shown_after_listen and "listening on" in line.lower():
                        print_banner(tunnel_url)
                        banner_shown_after_listen = True
            if engine_proc.poll() is not None:
                logger.error(
                    "Engine process terminated with code %d", engine_proc.returncode
                )
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        logger.info("Stopping engine and tunnel...")
    finally:
        httpd.server_close()
        engine_proc.terminate()
        if tunnel_proc:
            tunnel_proc.terminate()
        try:
            from pyngrok import ngrok  # type: ignore[import-untyped]

            ngrok.kill()
        except Exception:  # noqa: BLE001, S110
            pass


if __name__ == "__main__":
    main()
