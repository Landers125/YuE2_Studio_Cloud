"""YuE2 Studio external engine runner for Kaggle (Dual Tesla T4 GPUs).

Designed for Kaggle's 2x Tesla T4 GPU accelerator:
- GPU 0 (CUDA:0): 16 GB VRAM dedicated to yue-server (YuE2-3B-Q8_0 / BF16).
- GPU 1 (CUDA:1): 16 GB VRAM dedicated to MuScriptor (Audio-to-MIDI transcription).
- Zero VRAM contention and instant parallel execution.
- Auto-detects /kaggle/input datasets to reuse pre-uploaded models with zero download time.
- Cloudflare Tunnel and ngrok support with kaggle_secrets integration.
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
import urllib.parse
import urllib.request
import uuid
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
logger: logging.Logger = logging.getLogger("yue2-kaggle-2xt4")

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
            size_mb=2820,
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
            size_mb=726,
            role="transcriber",
        ),
    ],
    "q5_k_m": [
        ModelFileConfig(
            filename="YuE2-3B-Q5_K_M.gguf",
            url=f"https://huggingface.co/{HF_REPO_YUE_GGUF}/resolve/{HF_REV_YUE_GGUF}/YuE2-3B-Q5_K_M.gguf",
            size_mb=2480,
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
            size_mb=640,
            role="transcriber",
        ),
    ],
}

HF_REPO_ACESTEP_GGUF: Final[str] = "Serveurperso/ACE-Step-1.5-GGUF"
HF_REV_ACESTEP_GGUF: Final[str] = "666ac70204440867d8c01ba4b119cc79c95b370a"

ACESTEP_MODEL_CATALOG: Final[dict[str, list[ModelFileConfig]]] = {
    "xl_turbo_q8": [
        ModelFileConfig(
            filename="acestep-v15-xl-turbo-Q8_0.gguf",
            url=f"https://huggingface.co/{HF_REPO_ACESTEP_GGUF}/resolve/{HF_REV_ACESTEP_GGUF}/acestep-v15-xl-turbo-Q8_0.gguf",
            size_mb=5305,
            role="dit",
        ),
        ModelFileConfig(
            filename="acestep-5Hz-lm-1.7B-Q8_0.gguf",
            url=f"https://huggingface.co/{HF_REPO_ACESTEP_GGUF}/resolve/{HF_REV_ACESTEP_GGUF}/acestep-5Hz-lm-1.7B-Q8_0.gguf",
            size_mb=1975,
            role="lm",
        ),
        ModelFileConfig(
            filename="Qwen3-Embedding-0.6B-Q8_0.gguf",
            url=f"https://huggingface.co/{HF_REPO_ACESTEP_GGUF}/resolve/{HF_REV_ACESTEP_GGUF}/Qwen3-Embedding-0.6B-Q8_0.gguf",
            size_mb=784,
            role="text_encoder",
        ),
        ModelFileConfig(
            filename="vae-BF16.gguf",
            url=f"https://huggingface.co/{HF_REPO_ACESTEP_GGUF}/resolve/{HF_REV_ACESTEP_GGUF}/vae-BF16.gguf",
            size_mb=337,
            role="vae",
        ),
    ],
    "xl_turbo_q4": [
        ModelFileConfig(
            filename="acestep-v15-xl-turbo-Q4_K_M.gguf",
            url=f"https://huggingface.co/{HF_REPO_ACESTEP_GGUF}/resolve/{HF_REV_ACESTEP_GGUF}/acestep-v15-xl-turbo-Q4_K_M.gguf",
            size_mb=2989,
            role="dit",
        ),
        ModelFileConfig(
            filename="acestep-5Hz-lm-1.7B-Q8_0.gguf",
            url=f"https://huggingface.co/{HF_REPO_ACESTEP_GGUF}/resolve/{HF_REV_ACESTEP_GGUF}/acestep-5Hz-lm-1.7B-Q8_0.gguf",
            size_mb=1975,
            role="lm",
        ),
        ModelFileConfig(
            filename="Qwen3-Embedding-0.6B-Q8_0.gguf",
            url=f"https://huggingface.co/{HF_REPO_ACESTEP_GGUF}/resolve/{HF_REV_ACESTEP_GGUF}/Qwen3-Embedding-0.6B-Q8_0.gguf",
            size_mb=784,
            role="text_encoder",
        ),
        ModelFileConfig(
            filename="vae-BF16.gguf",
            url=f"https://huggingface.co/{HF_REPO_ACESTEP_GGUF}/resolve/{HF_REV_ACESTEP_GGUF}/vae-BF16.gguf",
            size_mb=337,
            role="vae",
        ),
    ],
    "turbo_2b_q8": [
        ModelFileConfig(
            filename="acestep-v15-turbo-Q8_0.gguf",
            url=f"https://huggingface.co/{HF_REPO_ACESTEP_GGUF}/resolve/{HF_REV_ACESTEP_GGUF}/acestep-v15-turbo-Q8_0.gguf",
            size_mb=2550,
            role="dit",
        ),
        ModelFileConfig(
            filename="acestep-5Hz-lm-0.6B-Q8_0.gguf",
            url=f"https://huggingface.co/{HF_REPO_ACESTEP_GGUF}/resolve/{HF_REV_ACESTEP_GGUF}/acestep-5Hz-lm-0.6B-Q8_0.gguf",
            size_mb=710,
            role="lm",
        ),
        ModelFileConfig(
            filename="Qwen3-Embedding-0.6B-Q8_0.gguf",
            url=f"https://huggingface.co/{HF_REPO_ACESTEP_GGUF}/resolve/{HF_REV_ACESTEP_GGUF}/Qwen3-Embedding-0.6B-Q8_0.gguf",
            size_mb=784,
            role="text_encoder",
        ),
        ModelFileConfig(
            filename="vae-BF16.gguf",
            url=f"https://huggingface.co/{HF_REPO_ACESTEP_GGUF}/resolve/{HF_REV_ACESTEP_GGUF}/vae-BF16.gguf",
            size_mb=337,
            role="vae",
        ),
    ],
}


def detect_environment() -> tuple[Path, int]:
    """Detects working directory and available GPU count (Kaggle or Colab/Linux)."""
    kaggle_dir = Path("/kaggle/working")
    work_dir = kaggle_dir if kaggle_dir.exists() else Path("/content") if Path("/content").exists() else Path.cwd()

    gpu_count = 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        )
        lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
        gpu_count = len(lines)
        logger.info("Detected %d NVIDIA GPU(s):", gpu_count)
        for i, line in enumerate(lines):
            logger.info("  [GPU %d] %s", i, line)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not query GPUs via nvidia-smi: %s", exc)

    return work_dir, gpu_count


def ensure_system_packages(work_dir: Path) -> None:
    """Installs required system packages only if yue-server needs to be built from source."""
    # Fast path: if yue-server is already present locally or in dataset, skip heavy installs!
    if (work_dir / "yue-server").exists():
        return
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists() and any(kaggle_input.rglob("yue-server")):
        return

    packages = ["cmake", "build-essential", "aria2", "ninja-build", "git"]
    missing: list[str] = []
    for pkg in packages:
        check_binary = "aria2c" if pkg == "aria2" else pkg
        check_binary = "gcc" if pkg == "build-essential" else check_binary
        check_binary = "ninja" if pkg == "ninja-build" else check_binary
        if shutil.which(check_binary) is None:
            missing.append(pkg)

    if missing and shutil.which("apt-get"):
        logger.info("Installing build packages: %s", ", ".join(missing))
        try:
            subprocess.check_call(
                ["apt-get", "install", "-y", "-qq", "--no-install-recommends"] + missing
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            logger.warning("Package installation notice: %s", exc)


def is_binary_runnable(binary: Path) -> bool:
    """Verifies that a compiled binary can load its dynamic libraries without error 127."""
    if not binary.exists():
        return False
    work_dir = binary.parent
    env = os.environ.copy()
    cuda_paths = [
        str(work_dir),
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(cuda_paths) + ":" + env.get("LD_LIBRARY_PATH", "")
    try:
        res = subprocess.run(
            [str(binary), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            env=env,
        )
        if res.returncode == 127:
            err = res.stderr.decode(errors="ignore").strip()
            logger.warning("Binary %s is incompatible with current CUDA (error 127: %s)", binary.name, err)
            return False
        return True
    except subprocess.TimeoutExpired:
        # Dynamic loading succeeded, initialization taking time on multi-GPU
        return True
    except Exception as exc:
        logger.warning("Binary check notice: %s", exc)
        return False


def build_yue_engine(work_dir: Path) -> Path:
    """Builds yue2.cpp with CUDA sm_75 architecture for Tesla T4."""
    local_target = work_dir / "yue-server"
    local_libs = list(work_dir.glob("lib*.so*"))

    if local_target.exists() and local_libs:
        local_target.chmod(0o755)
        logger.info("Found cached working yue-server with libraries at %s", local_target)
        return local_target

    # Check /kaggle/input for pre-compiled yue-server dataset
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for candidate in kaggle_input.rglob("yue-server"):
            if candidate.is_file():
                cand_dir = candidate.parent
                cand_libs = list(cand_dir.glob("lib*.so*"))
                if cand_libs:
                    shutil.copy2(candidate, local_target)
                    local_target.chmod(0o755)
                    for lib in cand_libs:
                        shutil.copy2(lib, work_dir / lib.name)
                    # Adapt CUDA 13 soname/version to CUDA 12 if needed across all variants
                    for c_lib in work_dir.glob("libggml-cuda*"):
                        if c_lib.is_file():
                            try:
                                raw = c_lib.read_bytes()
                                if b"libcudart.so.13" in raw or b"libcublas.so.13" in raw:
                                    patched = raw.replace(b"libcudart.so.13", b"libcudart.so.12").replace(b"libcublas.so.13", b"libcublas.so.12")
                                    c_lib.write_bytes(patched)
                                    logger.info("Patched %s for native CUDA 12 compatibility.", c_lib.name)
                            except Exception as p_exc:
                                logger.warning("Could not patch %s: %s", c_lib.name, p_exc)
                    logger.info("Restoring pre-compiled yue-server from Kaggle Input: %s", candidate)
                    return local_target

    # Check if gdown placed yue-server in work_dir / "YuE2"
    gdown_dir = work_dir / "YuE2"
    if gdown_dir.exists():
        gdown_target = gdown_dir / "yue-server"
        gdown_libs = list(gdown_dir.glob("lib*.so*"))
        if gdown_target.exists() and gdown_libs:
            shutil.copy2(gdown_target, local_target)
            local_target.chmod(0o755)
            for lib in gdown_libs:
                shutil.copy2(lib, work_dir / lib.name)
            logger.info("Restored compatible yue-server from Google Drive (%s)...", gdown_dir)
            return local_target

    repo_url = "https://github.com/timoncool/yue2.cpp.git"
    source_dir = work_dir / "yue2.cpp"
    build_dir = source_dir / "build"

    # Ensure libcuda.so symlink exists in Kaggle container (required for CMake CUDA::cuda_driver target)
    for src in [
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/usr/local/cuda/lib64/stubs/libcuda.so"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so"),
    ]:
        if src.exists():
            for dst in [
                Path("/usr/lib/x86_64-linux-gnu/libcuda.so"),
                Path("/usr/local/cuda/lib64/libcuda.so"),
                Path("/usr/local/cuda/targets/x86_64-linux/lib/libcuda.so"),
            ]:
                if not dst.exists():
                    try:
                        dst.symlink_to(src)
                        logger.info("Created CUDA driver symlink: %s -> %s", dst, src)
                    except Exception:
                        pass
            break

    # Clean prior failed build directory
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)

    if not source_dir.exists():
        logger.info("Cloning yue2.cpp repository...")
        subprocess.check_call(
            ["git", "clone", "--recurse-submodules", "-q", repo_url, str(source_dir)]
        )

    logger.info("Configuring CMake for Tesla T4 (Turing sm_75)...")
    cmake_cmd = [
        "cmake",
        "-B",
        str(build_dir),
        "-S",
        str(source_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA=ON",
        "-DGGML_CUDA_NO_VMM=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=75",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    ]
    for stub_dir in [
        "/usr/local/cuda/lib64/stubs",
        "/usr/local/cuda/targets/x86_64-linux/lib/stubs",
        "/usr/lib/x86_64-linux-gnu",
    ]:
        if Path(stub_dir).exists():
            cmake_cmd.append(f"-DCMAKE_LIBRARY_PATH={stub_dir}")
            break

    if shutil.which("ninja"):
        cmake_cmd.extend(["-G", "Ninja"])
    subprocess.check_call(cmake_cmd)

    cpu_cores = os.cpu_count() or 4
    logger.info("Compiling yue2.cpp with %d CPU threads...", cpu_cores)
    subprocess.check_call(
        ["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(cpu_cores)]
    )

    built_binaries = list(build_dir.rglob("yue-server"))
    if not built_binaries:
        raise FileNotFoundError("Compilation finished but yue-server was not found.")

    target_exe = built_binaries[0]
    shutil.copy2(target_exe, local_target)
    local_target.chmod(0o755)
    logger.info("Successfully built yue-server: %s", local_target)

    # Collect and copy all shared libraries
    for f in build_dir.rglob("*.so*"):
        if f.is_file() and not f.is_symlink():
            try:
                shutil.copy2(f, work_dir / f.name)
            except Exception:
                pass
        elif f.is_symlink():
            try:
                shutil.copy2(f.resolve(), work_dir / f.name)
            except Exception:
                pass

    return local_target


def download_models(
    models_dir: Path, profile: str, include_transcriber: bool = True
) -> dict[str, Path]:
    """Downloads required GGUF weights into models_dir or finds them in /kaggle/input."""
    models_dir.mkdir(parents=True, exist_ok=True)
    configs = MODEL_CATALOG.get(profile, MODEL_CATALOG["q8_0"])
    has_aria2c = shutil.which("aria2c") is not None
    resolved: dict[str, Path] = {}

    kaggle_input = Path("/kaggle/input")

    for cfg in configs:
        if cfg.role == "transcriber" and not include_transcriber:
            continue

        min_bytes = int(cfg.size_mb * 1024 * 1024 * 0.9)

        # 1. Check if attached as a Kaggle Input Dataset
        if kaggle_input.exists():
            matches = list(kaggle_input.rglob(cfg.filename))
            valid = [m for m in matches if m.is_file() and m.stat().st_size > min_bytes]
            if valid:
                logger.info("Found '%s' in Kaggle input: %s", cfg.filename, valid[0])
                resolved[cfg.role] = valid[0]
                continue

        # 2. Check work_dir / "YuE2" (gdown directory)
        work_base = models_dir.parent
        gdown_candidates = [
            work_base / "YuE2" / cfg.filename,
            work_base / "YuE2" / "models" / cfg.filename,
        ]
        found_gdown = False
        for gc in gdown_candidates:
            if gc.exists() and gc.stat().st_size > min_bytes:
                logger.info("Found '%s' from Google Drive (%s)", cfg.filename, gc)
                resolved[cfg.role] = gc
                found_gdown = True
                break
        if found_gdown:
            continue

        # 3. Check local models dir
        dest = models_dir / cfg.filename
        resolved[cfg.role] = dest
        if dest.exists() and dest.stat().st_size > min_bytes:
            logger.info("Model '%s' already downloaded (%d MB).", cfg.filename, cfg.size_mb)
            continue

        # 3. Download via aria2c
        logger.info("Downloading %s (~%d MB) into %s...", cfg.filename, cfg.size_mb, models_dir)
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
                    str(models_dir),
                    "--out",
                    cfg.filename,
                    cfg.url,
                ]
            )
        else:
            urllib.request.urlretrieve(cfg.url, str(dest))
        logger.info("Downloaded %s successfully.", cfg.filename)

    return resolved


def build_acestep_engine(work_dir: Path) -> Path:
    """Builds acestep.cpp with CUDA sm_75 architecture for Tesla T4."""
    local_target = work_dir / "ace-server"
    local_libs = list(work_dir.glob("lib*.so*"))

    if local_target.exists() and local_libs:
        local_target.chmod(0o755)
        logger.info("Found cached working ace-server with libraries at %s", local_target)
        return local_target

    # Check /kaggle/input for pre-compiled ace-server dataset
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for candidate in kaggle_input.rglob("ace-server"):
            if candidate.is_file():
                cand_dir = candidate.parent
                cand_libs = list(cand_dir.glob("lib*.so*"))
                if cand_libs:
                    shutil.copy2(candidate, local_target)
                    local_target.chmod(0o755)
                    for lib in cand_libs:
                        shutil.copy2(lib, work_dir / lib.name)
                    logger.info("Restoring pre-compiled ace-server from Kaggle Input: %s", candidate)
                    return local_target

    repo_url = "https://github.com/timoncool/acestep.cpp.git"
    source_dir = work_dir / "acestep.cpp"
    build_dir = source_dir / "build"

    for src in [
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/usr/local/cuda/lib64/stubs/libcuda.so"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so"),
    ]:
        if src.exists():
            for dst in [
                Path("/usr/lib/x86_64-linux-gnu/libcuda.so"),
                Path("/usr/local/cuda/lib64/libcuda.so"),
                Path("/usr/local/cuda/targets/x86_64-linux/lib/libcuda.so"),
            ]:
                if not dst.exists():
                    try:
                        dst.symlink_to(src)
                    except Exception:
                        pass
            break

    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)

    if not source_dir.exists():
        logger.info("Cloning acestep.cpp repository...")
        subprocess.check_call(
            ["git", "clone", "--recurse-submodules", "-b", "studio", "-q", repo_url, str(source_dir)]
        )

    logger.info("Configuring CMake for Tesla T4 (Turing sm_75) for acestep.cpp...")
    cmake_cmd = [
        "cmake",
        "-B",
        str(build_dir),
        "-S",
        str(source_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA=ON",
        "-DGGML_CUDA_NO_VMM=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=75",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    ]
    for stub_dir in [
        "/usr/local/cuda/lib64/stubs",
        "/usr/local/cuda/targets/x86_64-linux/lib/stubs",
        "/usr/lib/x86_64-linux-gnu",
    ]:
        if Path(stub_dir).exists():
            cmake_cmd.append(f"-DCMAKE_LIBRARY_PATH={stub_dir}")
            break

    if shutil.which("ninja"):
        cmake_cmd.extend(["-G", "Ninja"])
    subprocess.check_call(cmake_cmd)

    cpu_cores = os.cpu_count() or 4
    logger.info("Compiling ace-server with %d CPU threads...", cpu_cores)
    subprocess.check_call(
        ["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(cpu_cores), "--target", "ace-server"]
    )

    built_binaries = list(build_dir.rglob("ace-server"))
    if not built_binaries:
        raise FileNotFoundError("Compilation finished but ace-server was not found.")

    target_exe = built_binaries[0]
    shutil.copy2(target_exe, local_target)
    local_target.chmod(0o755)
    logger.info("Successfully built ace-server: %s", local_target)

    for f in build_dir.rglob("*.so*"):
        if f.is_file() and not f.is_symlink():
            try:
                shutil.copy2(f, work_dir / f.name)
            except Exception:
                pass
        elif f.is_symlink():
            try:
                shutil.copy2(f.resolve(), work_dir / f.name)
            except Exception:
                pass

    return local_target


def download_acestep_models(
    models_dir: Path, variant: str = "xl_turbo_q8"
) -> dict[str, Path]:
    """Downloads required ACE-Step GGUF weights or resolves them from /kaggle/input."""
    models_dir.mkdir(parents=True, exist_ok=True)
    configs = ACESTEP_MODEL_CATALOG.get(variant, ACESTEP_MODEL_CATALOG["xl_turbo_q8"])
    resolved: dict[str, Path] = {}
    has_aria2c = shutil.which("aria2c") is not None

    kaggle_input = Path("/kaggle/input")
    for cfg in configs:
        target_file = models_dir / cfg.filename
        min_bytes = int(cfg.size_mb * 1024 * 1024 * 0.9)

        if target_file.exists() and target_file.stat().st_size >= min_bytes:
            logger.info("ACE-Step model '%s' already ready.", cfg.filename)
            resolved[cfg.role] = target_file
            continue

        if kaggle_input.exists():
            matches = list(kaggle_input.rglob(cfg.filename))
            valid = [m for m in matches if m.is_file() and m.stat().st_size >= min_bytes]
            if valid:
                found = valid[0]
                logger.info("Found '%s' in Kaggle dataset: %s", cfg.filename, found)
                try:
                    if target_file.is_symlink() or target_file.exists():
                        target_file.unlink(missing_ok=True)
                    target_file.symlink_to(found)
                    resolved[cfg.role] = target_file
                    continue
                except Exception as exc:
                    logger.warning("Symlink error (%s), using direct path %s", exc, found)
                    resolved[cfg.role] = found
                    continue

        logger.info("Downloading ACE-Step model %s (~%d MB)...", cfg.filename, cfg.size_mb)
        if has_aria2c:
            subprocess.check_call(
                [
                    "aria2c",
                    "-c",
                    "-x", "16",
                    "-s", "16",
                    "-k", "1M",
                    "--file-allocation=none",
                    "--dir", str(models_dir),
                    "--out", cfg.filename,
                    cfg.url,
                ]
            )
        else:
            urllib.request.urlretrieve(cfg.url, str(target_file))
        logger.info("Downloaded %s successfully.", cfg.filename)
        resolved[cfg.role] = target_file

    return resolved


def setup_cloudflare_tunnel(port: int, work_dir: Path) -> tuple[subprocess.Popen[str], str]:
    """Downloads and launches cloudflared tunnel returning the public HTTPS URL."""
    cloudflared_bin = work_dir / "cloudflared"
    if not cloudflared_bin.exists():
        logger.info("Downloading cloudflared binary...")
        urllib.request.urlretrieve(CLOUDFLARED_URL, str(cloudflared_bin))
        cloudflared_bin.chmod(0o755)

    cmd = [
        str(cloudflared_bin),
        "tunnel",
        "--url",
        f"http://127.0.0.1:{port}",
        "--no-autoupdate",
    ]
    logger.info("Launching Cloudflare tunnel on port %d...", port)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    public_url = ""
    start_time = time.time()
    while time.time() - start_time < 45:
        if proc.stdout:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            match = url_pattern.search(line)
            if match:
                public_url = match.group(0)
                break
        time.sleep(0.1)

    if not public_url:
        logger.warning("Could not automatically extract Cloudflare URL within 45s.")
        public_url = f"http://127.0.0.1:{port}"

    return proc, public_url


def setup_ngrok_tunnel(
    port: int, authtoken: str | None = None
) -> tuple[subprocess.Popen[str] | None, str]:
    """Configures and runs ngrok tunnel using pyngrok."""
    try:
        from pyngrok import conf, ngrok  # type: ignore[import-untyped]
    except ImportError:
        logger.info("Installing pyngrok...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyngrok"])
        from pyngrok import conf, ngrok  # type: ignore[import-untyped]

    token = authtoken or ""
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore[import-untyped]
            user_secrets = UserSecretsClient()
            token = user_secrets.get_secret("NGROK_TOKEN") or ""
        except Exception:
            pass

    if token:
        conf.get_default().auth_token = token

    try:
        ngrok.kill()
    except Exception:
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
    gpu_id: int = 0,
) -> subprocess.Popen[str]:
    """Runs the yue-server process on the designated GPU (GPU 0 on Kaggle 2xT4)."""
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
        logger.info("Configured YuE2 companion adapter: %s", models["companion"])
    cmd.append("--keep-loaded")

    if adapters_dir and adapters_dir.exists():
        has_adapters = any(p.is_dir() for p in adapters_dir.iterdir())
        if has_adapters:
            cmd.extend(["--adapters", str(adapters_dir)])
            logger.info("Configured LoRA adapters directory: %s", adapters_dir)

    logger.info("Executing yue-server on GPU %d (CUDA%d): %s", gpu_id, gpu_id, " ".join(cmd))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["GGML_BACKEND"] = "CUDA0"

    work_dir = executable.parent
    build_dir = work_dir / "yue2.cpp" / "build"
    cuda_paths = [
        str(work_dir),
        str(build_dir),
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(cuda_paths) + ":" + env.get("LD_LIBRARY_PATH", "")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    return process


def launch_acestep_engine(
    executable: Path,
    models_dir: Path,
    adapters_dir: Path | None = None,
    port: int = 18089,
    gpu_id: int = 0,
) -> subprocess.Popen[str]:
    """Runs ace-server process on designated GPU."""
    cmd = [
        str(executable),
        "--models",
        str(models_dir),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--keep-loaded",
    ]
    if adapters_dir and adapters_dir.exists():
        has_adapters = any(p.is_dir() for p in adapters_dir.iterdir())
        if has_adapters:
            cmd.extend(["--adapters", str(adapters_dir)])
            logger.info("Configured ACE-Step LoRA adapters: %s", adapters_dir)

    logger.info("Executing ace-server on GPU %d (CUDA%d): %s", gpu_id, gpu_id, " ".join(cmd))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["GGML_BACKEND"] = "CUDA0"

    work_dir = executable.parent
    build_dir = work_dir / "acestep.cpp" / "build"
    cuda_paths = [
        str(work_dir),
        str(build_dir),
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(cuda_paths) + ":" + env.get("LD_LIBRARY_PATH", "")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    return process


TRANSCRIBE_JOBS: dict[str, dict[str, object]] = {}


def ensure_muscriptor_model(work_dir: Path, size: str = "large") -> Path:
    """Ensures MuScriptor weights (small/medium/large) are downloaded or resolved from /kaggle/input."""
    size_norm = size.lower().strip()
    if "small" in size_norm or "103" in size_norm:
        key = "small"
    elif "medium" in size_norm or "307" in size_norm:
        key = "medium"
    else:
        key = "large"

    target_dir = work_dir / "models" / f"muscriptor-{key}"
    target_dir.mkdir(parents=True, exist_ok=True)
    safetensors = target_dir / "model.safetensors"
    config_json = target_dir / "config.json"

    specs = {
        "large": (
            "https://huggingface.co/cocktailpeanut/muscriptor-large/resolve/87f4bf981f56f90fb5043153b3f54af3c3053da9/model.safetensors",
            "https://huggingface.co/cocktailpeanut/muscriptor-large/resolve/87f4bf981f56f90fb5043153b3f54af3c3053da9/config.json",
            4_500_000_000,
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

    # 1. Check local models dir
    if safetensors.exists() and safetensors.stat().st_size >= min_bytes:
        logger.info("MuScriptor %s weights ready at %s", key, safetensors)
        return safetensors

    # 2. Check /kaggle/input for pre-uploaded Kaggle datasets (muscriptor-models or yue2-q8-models)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates = list(kaggle_input.rglob("model.safetensors"))
        valid = [c for c in candidates if c.is_file() and c.stat().st_size >= min_bytes]
        if valid:
            input_st = valid[0]
            input_cfg = input_st.parent / "config.json"
            logger.info(
                "Found pre-uploaded MuScriptor %s weights in Kaggle dataset: %s (%d MB)",
                key,
                input_st,
                input_st.stat().st_size // (1024 * 1024),
            )
            try:
                if safetensors.is_symlink() or safetensors.exists():
                    safetensors.unlink(missing_ok=True)
                safetensors.symlink_to(input_st)
                if input_cfg.exists():
                    if config_json.is_symlink() or config_json.exists():
                        config_json.unlink(missing_ok=True)
                    config_json.symlink_to(input_cfg)
                logger.info(
                    "Successfully symlinked MuScriptor weights into %s (0s load time)",
                    target_dir,
                )
                return safetensors
            except Exception as exc:
                logger.warning(
                    "Symlink failed (%s), using direct path %s", exc, input_st
                )
                return input_st

    # 3. Fallback: fast download via aria2c or urllib
    logger.info("Downloading MuScriptor %s weights into %s...", key, target_dir)
    has_aria2 = shutil.which("aria2c") is not None
    if not has_aria2:
        try:
            subprocess.run(["apt-get", "update", "-qq"], capture_output=True, timeout=30)
            subprocess.run(["apt-get", "install", "-y", "-qq", "aria2"], capture_output=True, timeout=30)
            has_aria2 = shutil.which("aria2c") is not None
        except Exception:
            pass

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


ACTIVE_JOB_ROUTES: dict[str, int] = {}


class UnifiedServerHandler(http.server.BaseHTTPRequestHandler):
    """Unified HTTP handler: routes between yue-server, ace-server, and MuScriptor."""

    backend_port: int = 18088  # yue-server
    ace_port: int = 18089      # ace-server
    work_dir: Path = Path("/kaggle/working")
    transcribe_gpu_id: int = 1

    def do_GET(self) -> None:
        if self.path in ("/health", "/ping"):
            target_engine = self.headers.get("X-Engine", "").lower()
            if not target_engine:
                parsed = urllib.parse.urlparse(self.path)
                qs = urllib.parse.parse_qs(parsed.query)
                target_engine = qs.get("engine", [""])[0].lower()

            if target_engine == "acestep":
                self._proxy_to_port("GET", self.ace_port)
                return
            if target_engine == "yue2":
                self._proxy_to_port("GET", self.backend_port)
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            resp_bytes = b'{"status":"ok"}'
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
            return
        if self.path in ("/logs", "/log"):
            target_engine = self.headers.get("X-Engine", "").lower()
            if not target_engine:
                parsed = urllib.parse.urlparse(self.path)
                qs = urllib.parse.parse_qs(parsed.query)
                target_engine = qs.get("engine", [""])[0].lower()

            if target_engine == "acestep":
                self._proxy_to_port("GET", self.ace_port)
                return
            if target_engine == "yue2":
                self._proxy_to_port("GET", self.backend_port)
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/shutdown":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Shutting down Kaggle instance...")
            logger.info("Shutdown requested via /shutdown. Stopping server...")
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0))).start()
            return
        if self.path.startswith("/transcribe"):
            self._handle_transcribe_get()
            return
        if self.path == "/props":
            self._handle_props_get()
            return
        if self.path.startswith("/job"):
            self._handle_job_get()
            return
        self._proxy("GET")

    def do_POST(self) -> None:
        if self.path == "/shutdown":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Shutting down Kaggle instance...")
            logger.info("Shutdown requested via /shutdown. Stopping server...")
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0))).start()
            return
        if self.path.startswith("/transcribe"):
            self._handle_transcribe_post()
            return
        if self.path == "/synth":
            self._handle_synth_post()
            return
        if self.path in ("/lm", "/understand", "/vae"):
            self._proxy_to_port("POST", self.ace_port)
            return
        self._proxy("POST")

    def do_HEAD(self) -> None:
        self._proxy("HEAD")

    def _handle_props_get(self) -> None:
        target_engine = self.headers.get("X-Engine", "").lower()
        if not target_engine:
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            target_engine = qs.get("engine", [""])[0].lower()

        if target_engine == "acestep":
            self._proxy_to_port("GET", self.ace_port)
            return
        if target_engine == "yue2":
            self._proxy_to_port("GET", self.backend_port)
            return

        props = {
            "version": "YuE2 & ACE-Step 1.5 Studio (2x Tesla T4)",
            "engines": ["yue2", "acestep"],
            "models": {
                "yue2": "YuE2-3B-Q8_0",
                "acestep": "acestep-v15-xl-turbo-Q8_0",
            },
            "sample_rate": 48000,
        }
        resp_bytes = json.dumps(props).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def _handle_synth_post(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        target_engine = self.headers.get("X-Engine", "").lower()
        target_port = self.backend_port
        if target_engine == "acestep":
            target_port = self.ace_port

        try:
            req_json = json.loads(body.decode("utf-8"))
            engine_req = str(req_json.get("engine", "")).lower()
            if target_engine == "acestep" or engine_req == "acestep" or "caption" in req_json or "task_type" in req_json or "bpm" in req_json:
                target_port = self.ace_port
                if "caption" not in req_json and "style" in req_json:
                    req_json["caption"] = req_json.pop("style")
                if "task_type" not in req_json:
                    req_json["task_type"] = "text2music"
                if "inference_steps" not in req_json and "steps" in req_json:
                    req_json["inference_steps"] = req_json.pop("steps")
                req_json.pop("engine", None)
                body = json.dumps(req_json).encode("utf-8")
            else:
                req_json.pop("engine", None)
                body = json.dumps(req_json).encode("utf-8")
        except Exception:
            pass

        req_headers: dict[str, str] = {}
        for header, value in self.headers.items():
            if header.lower() not in ("host", "content-length"):
                req_headers[header] = value
        req_headers["Content-Length"] = str(len(body))

        target = f"http://127.0.0.1:{target_port}/synth"
        req = urllib.request.Request(target, data=body, headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                self.send_response(resp.status)
                for header, value in resp.headers.items():
                    if header.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(header, value)
                resp_data = resp.read()
                try:
                    res_json = json.loads(resp_data.decode("utf-8"))
                    job_id = res_json.get("id") or res_json.get("job_id")
                    if job_id:
                        ACTIVE_JOB_ROUTES[str(job_id)] = target_port
                except Exception:
                    pass
                self.send_header("Content-Length", str(len(resp_data)))
                self.end_headers()
                self.wfile.write(resp_data)
        except urllib.error.HTTPError as exc:
            self.send_response(exc.code)
            self.end_headers()
            self.wfile.write(exc.read())
        except Exception as exc:  # noqa: BLE001
            logger.error("Synth proxy error (port %d): %s", target_port, exc)
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(exc).encode())

    def _handle_job_get(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        job_id = qs.get("id", [""])[0].strip()
        target_port = ACTIVE_JOB_ROUTES.get(job_id)

        if target_port:
            self._proxy_to_port("GET", target_port)
            return

        target = f"http://127.0.0.1:{self.backend_port}{self.path}"
        req = urllib.request.Request(target, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                self.send_response(resp.status)
                for header, value in resp.headers.items():
                    if header.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(header, value)
                resp_data = resp.read()
                self.send_header("Content-Length", str(len(resp_data)))
                self.end_headers()
                self.wfile.write(resp_data)
                return
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                self._proxy_to_port("GET", self.ace_port)
                return
            self.send_response(exc.code)
            self.end_headers()
            self.wfile.write(exc.read())
        except Exception:
            self._proxy_to_port("GET", self.ace_port)

    def _proxy(self, method: str) -> None:
        target_engine = self.headers.get("X-Engine", "").lower()
        if not target_engine:
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            target_engine = qs.get("engine", [""])[0].lower()

        if target_engine == "acestep":
            self._proxy_to_port(method, self.ace_port)
        else:
            self._proxy_to_port(method, self.backend_port)

    def _proxy_to_port(self, method: str, port: int) -> None:
        target = f"http://127.0.0.1:{port}{self.path}"
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
            logger.error("Internal proxy error (port %d): %s", port, exc)
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(exc).encode())

    def _handle_transcribe_get(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            job_id = qs.get("id", [""])[0].strip()
            if not job_id:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "Missing id parameter"}')
                return

            job = TRANSCRIBE_JOBS.get(job_id)
            if not job:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "Job not found"}')
                return

            is_result = "result" in qs or qs.get("result", ["0"])[0] in ("1", "true", "yes")
            if is_result:
                out_path = job.get("output")
                if not isinstance(out_path, Path) or not out_path.exists():
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": "Result file not found or not ready"}')
                    return

                midi_bytes = out_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "audio/midi")
                self.send_header("Content-Disposition", f'attachment; filename="{job_id}.mid"')
                self.send_header("Content-Length", str(len(midi_bytes)))
                self.end_headers()
                self.wfile.write(midi_bytes)
                return

            status = job.get("status", "running")
            start_t = float(job.get("start_time", time.time()))
            elapsed = round(time.time() - start_t, 1)
            payload = {
                "job_id": job_id,
                "status": status,
                "elapsed": elapsed,
                "error": job.get("error"),
            }
            resp_bytes = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.error("Transcribe GET error: %s", exc)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(exc).encode())

    def _handle_transcribe_post(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length <= 0:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Empty audio payload")
                return

            model_size = self.headers.get("X-Model-Size", "large").strip().lower()
            is_async = (
                self.headers.get("X-Async", "").strip() in ("1", "true")
                or "async=1" in self.path
                or "async=true" in self.path
            )

            audio_data = self.rfile.read(content_length)
            job_id = f"transcribe_{uuid.uuid4().hex[:12]}"
            tmp_in = Path(f"/tmp/in_{job_id}.mp3")
            tmp_out = Path(f"/tmp/out_{job_id}.mid")
            tmp_in.write_bytes(audio_data)

            def worker() -> None:
                try:
                    if shutil.which("muscriptor") is None:
                        logger.info("Installing muscriptor package...")
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", "-q", "muscriptor"]
                        )

                    model_path = ensure_muscriptor_model(self.work_dir, size=model_size)
                    logger.info(
                        "Transcribing audio (job=%s, %d bytes, model=%s) via MuScriptor on GPU %d...",
                        job_id,
                        len(audio_data),
                        model_size,
                        self.transcribe_gpu_id,
                    )

                    # Convert input MP3 to WAV for 100% reliable soundfile reading
                    tmp_wav = tmp_in.with_suffix(".wav")
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", str(tmp_in), "-ar", "44100", "-ac", "2", str(tmp_wav)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=30,
                        )
                    except Exception as ff_exc:
                        logger.warning("FFmpeg conversion notice: %s", ff_exc)

                    input_file = tmp_wav if (tmp_wav.exists() and tmp_wav.stat().st_size > 0) else tmp_in

                    cmd = [
                        "muscriptor",
                        "transcribe",
                        str(input_file),
                        "-o",
                        str(tmp_out),
                        "--model",
                        str(model_path),
                        "--device",
                        "cuda",
                        "--dtype",
                        "float16",
                    ]
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(self.transcribe_gpu_id)

                    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
                    if res.returncode != 0:
                        err_text = (res.stderr or res.stdout or "").strip()
                        logger.warning(
                            "MuScriptor GPU %d attempt failed (code %d): %s. Retrying on CPU...",
                            self.transcribe_gpu_id,
                            res.returncode,
                            err_text,
                        )
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
                        res = subprocess.run(cmd_cpu, capture_output=True, text=True)

                    if tmp_out.exists() and tmp_out.stat().st_size > 0:
                        TRANSCRIBE_JOBS[job_id]["status"] = "finished"
                        TRANSCRIBE_JOBS[job_id]["output"] = tmp_out
                        logger.info(
                            "Job %s finished successfully (%d bytes MIDI produced)",
                            job_id,
                            tmp_out.stat().st_size,
                        )
                    else:
                        TRANSCRIBE_JOBS[job_id]["status"] = "error"
                        TRANSCRIBE_JOBS[job_id]["error"] = (
                            res.stderr or res.stdout or "Transcription produced no output"
                        )
                        logger.error(
                            "Job %s failed: %s", job_id, TRANSCRIBE_JOBS[job_id]["error"]
                        )
                except Exception as exc:  # noqa: BLE001
                    TRANSCRIBE_JOBS[job_id]["status"] = "error"
                    TRANSCRIBE_JOBS[job_id]["error"] = str(exc)
                    logger.exception("Job %s crashed: %s", job_id, exc)

            TRANSCRIBE_JOBS[job_id] = {
                "status": "running",
                "output": None,
                "error": None,
                "start_time": time.time(),
            }

            if is_async:
                threading.Thread(target=worker, daemon=True).start()
                resp_data = json.dumps({"job_id": job_id, "status": "running"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_data)))
                self.end_headers()
                self.wfile.write(resp_data)
                return

            worker()
            if TRANSCRIBE_JOBS[job_id]["status"] == "finished" and tmp_out.exists():
                midi_data = tmp_out.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "audio/midi")
                self.send_header("Content-Disposition", 'attachment; filename="transcribed.mid"')
                self.send_header("Content-Length", str(len(midi_data)))
                self.end_headers()
                self.wfile.write(midi_data)
                if tmp_in.exists():
                    tmp_in.unlink(missing_ok=True)
                if tmp_out.exists():
                    tmp_out.unlink(missing_ok=True)
            else:
                err = TRANSCRIBE_JOBS[job_id].get("error", "Transcription failed on server")
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(f"Transcription failed: {err}".encode())
        except Exception as exc:  # noqa: BLE001
            logger.error("Transcribe POST error: %s", exc)
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


def print_banner(tunnel_url: str, gpu_count: int) -> None:
    """Prints user instructions with clickable banner and HTML card."""
    line = "=" * 70
    msg = (
        f"\n{line}\n"
        f"🚀 YuE2 & ACE-STEP 1.5 DUAL-ENGINE KAGGLE SERVER (2x TESLA T4)!\n"
        f"{line}\n"
        f"🎮 Аппаратное распределение:\n"
        f"   • GPU 0 (Tesla T4 #1, 16 GB): YuE2 Q8_0 & ACE-Step 1.5 Turbo (Синтез музыки)\n"
        f"   • GPU 1 (Tesla T4 #2, 16 GB): Транскрибация Аудио в MIDI (MuScriptor Large)\n"
        f"{line}\n"
        f"🔗 ССЫЛКА ДЛЯ connect-colab.bat / ANDROID / PC:\n"
        f"   >>> {tunnel_url} <<<\n"
        f"{line}\n"
        f"📋 Доступные модели:\n"
        f"  1. ⚡ ACE-Step 1.5 Turbo — сверхбыстрый синтез (~30 сек)\n"
        f"  2. 🎼 YuE2 Q8_0 — глубокая 3B модель с партитурой (~8 мин)\n"
        f"  3. 🎹 MuScriptor Large — транскрибация аудио в 5-дорожечный MIDI\n"
        f"{line}\n"
    )
    print(msg, flush=True)


def notify_ready(tunnel_url: str, topic: str | None = None) -> bool:
    """Publishes the tunnel URL across multiple channels (ntfy.envs.net, cl1p.net, ntfy.sh) for instant DPI-proof discovery."""
    import hashlib
    import urllib.parse
    import urllib.request

    topics: list[str] = []
    if topic:
        topics.append(topic)
    else:
        for u in [
            os.environ.get("KAGGLE_USERNAME", ""),
            "alexeyklochkov",
            "curtrassel",
        ]:
            if u:
                h = hashlib.sha256(u.strip().lower().encode("utf-8")).hexdigest()[:16]
                t = f"yue2_studio_{h}"
                if t not in topics:
                    topics.append(t)

    success = False
    for t in topics:
        # Channel 1: ntfy.envs.net (unblocked, DPI-proof ntfy instance)
        try:
            req = urllib.request.Request(
                f"https://ntfy.envs.net/{t}",
                data=tunnel_url.encode("utf-8"),
                headers={
                    "Title": "YuE2 & ACE-Step 1.5 Ready (2x Tesla T4)",
                    "Tags": "musical_note,rocket,zap",
                    "Priority": "high",
                },
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("📡 Published tunnel URL to ntfy.envs.net: %s", t)
                    success = True
        except Exception as exc:
            logger.warning("Could not publish to ntfy.envs.net (%s): %s", t, exc)

        # Channel 2: cl1p.net (internet clipboard, works seamlessly across all ISPs)
        try:
            form_data = urllib.parse.urlencode({"content": tunnel_url}).encode("utf-8")
            req = urllib.request.Request(f"https://cl1p.net/{t}", data=form_data)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("📡 Published tunnel URL to cl1p.net: %s", t)
                    success = True
        except Exception as exc:
            logger.warning("Could not publish to cl1p.net (%s): %s", t, exc)

        # Channel 3: ntfy.sh (fallback standard channel)
        try:
            req = urllib.request.Request(
                f"https://ntfy.sh/{t}",
                data=tunnel_url.encode("utf-8"),
                headers={
                    "Title": "YuE2 & ACE-Step 1.5 Ready (2x Tesla T4)",
                    "Tags": "musical_note,rocket,zap",
                    "Priority": "high",
                },
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("📡 Published tunnel URL to ntfy.sh: %s", t)
                    success = True
        except Exception as exc:
            logger.warning("Could not publish to ntfy.sh (%s): %s", t, exc)

    return success


def run_kaggle_server(
    model: str = "q8_0",
    engine: str = "dual",
    ace_variant: str = "xl_turbo_q8",
    port: int = DEFAULT_PORT,
    tunnel: str = "cloudflare",
    ngrok_token: str = "",
    keep_loaded: bool = True,
    no_transcriber: bool = False,
) -> None:
    """Initializes and runs the dual-GPU server on Kaggle."""
    work_dir, gpu_count = detect_environment()
    work_dir.mkdir(parents=True, exist_ok=True)
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    adapters_dir = work_dir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    # Determine GPU roles (Dual Tesla T4 isolation)
    yue_gpu_id = 0
    ace_gpu_id = 1 if gpu_count >= 2 else 0
    transcribe_gpu_id = 1 if gpu_count >= 2 else 0

    yue_port = port + 1
    ace_port = port + 2

    # Cleanup stale processes
    try:
        subprocess.call(["killall", "-9", "yue-server", "ace-server", "cloudflared"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
    except Exception:
        pass

    ensure_system_packages(work_dir)

    yue_proc: subprocess.Popen[str] | None = None
    ace_proc: subprocess.Popen[str] | None = None

    # 1. Start YuE2 engine if requested
    if engine in ("dual", "yue2", "auto"):
        try:
            engine_exe = build_yue_engine(work_dir)
            models = download_models(models_dir, model.lower(), include_transcriber=not no_transcriber)
            yue_proc = launch_engine(
                engine_exe,
                models,
                adapters_dir=adapters_dir,
                port=yue_port,
                gpu_id=yue_gpu_id,
            )
            logger.info("YuE2 engine started on port %d (GPU %d).", yue_port, yue_gpu_id)
        except Exception as exc:
            logger.error("Failed to launch yue-server: %s", exc)

    # 2. Start ACE-Step engine if requested
    if engine in ("dual", "acestep", "auto"):
        try:
            ace_exe = build_acestep_engine(work_dir)
            ace_models = download_acestep_models(models_dir, variant=ace_variant)
            ace_proc = launch_acestep_engine(
                ace_exe,
                models_dir=models_dir,
                adapters_dir=adapters_dir,
                port=ace_port,
                gpu_id=ace_gpu_id,
            )
            logger.info("ACE-Step engine started on port %d (GPU %d).", ace_port, ace_gpu_id)
        except Exception as exc:
            logger.error("Failed to launch ace-server: %s", exc)

    # Start Unified HTTP Server on primary port
    UnifiedServerHandler.backend_port = yue_port
    UnifiedServerHandler.ace_port = ace_port
    UnifiedServerHandler.work_dir = work_dir
    UnifiedServerHandler.transcribe_gpu_id = transcribe_gpu_id

    httpd = ReusableServer(("0.0.0.0", port), UnifiedServerHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    tunnel_proc: subprocess.Popen[str] | None = None
    tunnel_url = f"http://127.0.0.1:{port}"

    if tunnel == "cloudflare":
        tunnel_proc, tunnel_url = setup_cloudflare_tunnel(port, work_dir)
    elif tunnel == "ngrok":
        tunnel_proc, tunnel_url = setup_ngrok_tunnel(port, ngrok_token or None)

    print_banner(tunnel_url, gpu_count)
    notify_ready(tunnel_url)

    try:
        while True:
            for p in (yue_proc, ace_proc):
                if p and p.stdout:
                    line = p.stdout.readline()
                    if line:
                        sys.stdout.write(line)
                        sys.stdout.flush()
            time.sleep(0.01)
    except KeyboardInterrupt:
        logger.info("Stopping engines and tunnel...")
    finally:
        httpd.server_close()
        if yue_proc:
            yue_proc.terminate()
        if ace_proc:
            ace_proc.terminate()
        if tunnel_proc:
            tunnel_proc.terminate()
        try:
            from pyngrok import ngrok  # type: ignore[import-untyped]
            ngrok.kill()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="YuE2 & ACE-Step 1.5 Studio Kaggle Dual Server")
    parser.add_argument("--engine", default="dual", choices=["dual", "acestep", "yue2", "auto"])
    parser.add_argument("--model", default="q8_0", choices=["q8_0", "bf16", "q6_k", "q5_k_m"])
    parser.add_argument("--ace-variant", default="xl_turbo_q8", choices=["xl_turbo_q8", "xl_turbo_q4", "turbo_2b_q8"])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--tunnel", default="cloudflare", choices=["cloudflare", "ngrok"])
    parser.add_argument("--ngrok-token", default="")
    parser.add_argument("--keep-loaded", action="store_true", default=True)
    parser.add_argument("--no-transcriber", action="store_true", default=False)
    args = parser.parse_args()

    run_kaggle_server(
        model=args.model,
        engine=args.engine,
        ace_variant=args.ace_variant,
        port=args.port,
        tunnel=args.tunnel,
        ngrok_token=args.ngrok_token,
        keep_loaded=args.keep_loaded,
        no_transcriber=args.no_transcriber,
    )


if __name__ == "__main__":
    main()
