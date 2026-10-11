"""YuE2 Studio Dedicated External Engine Runner for Kaggle (Tesla T4 GPU).

Dedicated specifically to YuE2 (3B Q8_0 Dual-Stage LM + VAE + Flow-Matching).
Restores precompiled yue-server and models from /kaggle/input (0s download time).
Establishes Cloudflare tunnel and announces URL to YuE2 Studio applications.
"""

from __future__ import annotations

import collections
import hashlib
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
from pathlib import Path
from typing import Final

START_TIME: Final[float] = time.time()
LOG_BUFFER: collections.deque[str] = collections.deque(maxlen=300)


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


class FlushingStreamHandler(logging.StreamHandler):
    """Flushes output stream immediately after each log record to avoid Kaggle buffering."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class BufferLogHandler(logging.Handler):
    """Stores the latest log messages in an in-memory ring buffer for live monitoring."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            LOG_BUFFER.append(msg)
        except Exception:
            pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[FlushingStreamHandler(sys.stdout)],
    force=True,
)
logger: logging.Logger = logging.getLogger("yue2-server")

_buf_handler = BufferLogHandler()
_buf_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
)
logger.addHandler(_buf_handler)
logging.getLogger().addHandler(_buf_handler)

DEFAULT_PORT: Final[int] = 18087
YUE_BACKEND_PORT: Final[int] = 18088
YUE_BACKEND_PORT_2: Final[int] = 18089
CLOUDFLARED_URL: Final[str] = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
)

ACTIVE_DUAL_JOBS: dict[str, tuple[str, str]] = {}
DUAL_GPU_ENABLED: bool = False



def detect_environment() -> tuple[Path, int]:
    """Detects working directory and available GPU count."""
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
    except Exception as exc:
        logger.warning("Could not query GPUs via nvidia-smi: %s", exc)

    return work_dir, gpu_count


def check_engine_supports_adapters(bin_path: Path) -> bool:
    """Verifies that yue-server binary supports --adapters CLI argument without executing it."""
    try:
        raw = bin_path.read_bytes()
        return b"--adapters" in raw or b"adapters" in raw
    except Exception:
        return False


def save_dataset_version(work_dir: Path, target_bin: Path) -> None:
    """Publishes updated yue-server binary with LoRA support to Kaggle dataset."""
    try:
        logger.info("Preparing dataset update for 'jungterdai/yue2-q8-models'...")
        k_dir = Path.home() / ".kaggle"
        k_dir.mkdir(parents=True, exist_ok=True)
        cred = {"username": "jungterdai", "key": "a9c86217f94f6a3477c246d4c556a579"}
        (k_dir / "kaggle.json").write_text(json.dumps(cred), encoding="utf-8")
        try:
            (k_dir / "kaggle.json").chmod(0o600)
        except Exception:
            pass
        os.environ["KAGGLE_USERNAME"] = cred["username"]
        os.environ["KAGGLE_KEY"] = cred["key"]

        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        upload_dir = work_dir / "dataset_upload"
        upload_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "id": "jungterdai/yue2-q8-models",
            "title": "YuE2 Q8 Models and Engine",
            "subtitle": "YuE2 Q8 Models and Native Engine with LoRA Adapters Support",
            "description": "Pre-quantized YuE2 Q8 GGUF weights with native C++/CUDA yue-server supporting LoRA adapters.",
            "resources": []
        }
        (upload_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Copy new binary and libraries
        shutil.copy2(target_bin, upload_dir / "yue-server")
        (upload_dir / "yue-server").chmod(0o755)
        for so in work_dir.glob("lib*.so*"):
            if so.is_file() and not so.is_symlink():
                shutil.copy2(so, upload_dir / so.name)

        # Retain model weights via symlink / copy from /kaggle/input
        kaggle_input = Path("/kaggle/input")
        for fname in ["YuE2-3B-Q8_0.gguf", "YuE2-Vae-F32.gguf"]:
            matches = list(kaggle_input.rglob(fname)) if kaggle_input.exists() else []
            valid = [m for m in matches if m.is_file() and m.stat().st_size > 1024 * 1024]
            if valid:
                dst = upload_dir / fname
                if not dst.exists():
                    try:
                        dst.symlink_to(valid[0])
                    except Exception:
                        shutil.copy2(valid[0], dst)

        logger.info("Uploading updated engine and models to dataset 'jungterdai/yue2-q8-models'...")
        api.dataset_create_version(
            str(upload_dir),
            version_notes="Build yue-server from adapters branch with native LoRA support (sm_75 T4)",
            dir_mode="skip",
            quiet=False,
        )
        logger.info("✓ Successfully updated Kaggle dataset jungterdai/yue2-q8-models with new LoRA engine!")
    except Exception as exc:
        logger.warning("Could not auto-update Kaggle dataset: %s", exc)


def restore_yue_engine(work_dir: Path) -> Path:
    """Restores pre-compiled yue-server from dataset or compiles from source with LoRA support."""
    target_bin = work_dir / "yue-server"
    local_libs = list(work_dir.glob("lib*.so*"))

    if target_bin.exists() and local_libs and check_engine_supports_adapters(target_bin):
        target_bin.chmod(0o755)
        logger.info("Cached yue-server with LoRA support found at %s", target_bin)
        return target_bin

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for candidate in kaggle_input.rglob("yue-server"):
            if candidate.is_file() and check_engine_supports_adapters(candidate):
                cand_dir = candidate.parent
                cand_libs = list(cand_dir.glob("lib*.so*"))
                if cand_libs:
                    shutil.copy2(candidate, target_bin)
                    target_bin.chmod(0o755)
                    for lib in cand_libs:
                        shutil.copy2(lib, work_dir / lib.name)
                    # Patch CUDA 13 soname to CUDA 12 if needed
                    for c_lib in work_dir.glob("libggml-cuda*"):
                        if c_lib.is_file():
                            try:
                                raw = c_lib.read_bytes()
                                if b"libcudart.so.13" in raw or b"libcublas.so.13" in raw:
                                    patched = raw.replace(b"libcudart.so.13", b"libcudart.so.12").replace(b"libcublas.so.13", b"libcublas.so.12")
                                    c_lib.write_bytes(patched)
                                    logger.info("Patched %s for CUDA 12 compatibility.", c_lib.name)
                            except Exception as p_exc:
                                logger.warning("Could not patch %s: %s", c_lib.name, p_exc)
                    logger.info("Restored yue-server with LoRA support from Kaggle Input: %s", candidate)
                    return target_bin

    logger.info("Pre-compiled yue-server with LoRA support not found in dataset. Building from 'adapters' branch...")
    repo_url = "https://github.com/timoncool/yue2.cpp.git"
    source_dir = work_dir / "yue2.cpp"
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
        logger.info("Cloning yue2.cpp repository (adapters branch with native LoRA support)...")
        subprocess.check_call(
            ["git", "clone", "--recurse-submodules", "-b", "adapters", "-q", repo_url, str(source_dir)]
        )

    logger.info("Configuring CMake for Tesla T4 (Turing sm_75)...")
    cmake_cmd = [
        "cmake",
        "-B", str(build_dir),
        "-S", str(source_dir),
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
    shutil.copy2(target_exe, target_bin)
    target_bin.chmod(0o755)
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

    logger.info("Successfully built yue-server with LoRA support: %s", target_bin)
    save_dataset_version(work_dir, target_bin)
    return target_bin


def resolve_yue_models(work_dir: Path) -> dict[str, Path]:
    """Finds required GGUF weights in /kaggle/input or downloads from HuggingFace."""
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    kaggle_input = Path("/kaggle/input")

    download_specs = {
        "backbone": (
            "YuE2-3B-Q8_0.gguf",
            "https://huggingface.co/Serveurperso/YuE2-GGUF/resolve/64b030e3deb6e8150d2b7c0db641ef5a17eca8a3/YuE2-3B-Q8_0.gguf",
        ),
        "vae": (
            "YuE2-Vae-F32.gguf",
            "https://huggingface.co/Serveurperso/YuE2-GGUF/resolve/64b030e3deb6e8150d2b7c0db641ef5a17eca8a3/YuE2-Vae-F32.gguf",
        ),
    }
    resolved: dict[str, Path] = {}

    for role, (filename, url) in download_specs.items():
        matches = list(kaggle_input.rglob(filename)) if kaggle_input.exists() else []
        valid = [m for m in matches if m.is_file() and m.stat().st_size > 1024 * 1024]
        if valid:
            logger.info("Found '%s' in Kaggle dataset: %s", filename, valid[0])
            resolved[role] = valid[0]
            continue

        dest = models_dir / filename
        if dest.exists() and dest.stat().st_size > 1024 * 1024:
            logger.info("Model '%s' already cached: %s", filename, dest)
            resolved[role] = dest
            continue

        logger.info("Downloading %s from Hugging Face...", filename)
        if shutil.which("aria2c"):
            subprocess.check_call(
                [
                    "aria2c",
                    "-c",
                    "-x", "16",
                    "-s", "16",
                    "-k", "1M",
                    "--file-allocation=none",
                    "--dir", str(models_dir),
                    "--out", filename,
                    url,
                ]
            )
        else:
            urllib.request.urlretrieve(url, str(dest))
        logger.info("Downloaded %s successfully.", filename)
        resolved[role] = dest

    return resolved


LORA_CATALOG: Final[dict[str, list[tuple[str, str]]]] = {
    "yue2-instrumental": [
        ("https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras/resolve/main/ar_lora_inst_v3abc.bf16.safetensors", "ar_lora_inst_v3abc.bf16.safetensors"),
    ],
    "yue2-canzone-italiana": [
        ("https://huggingface.co/becausereasons/yue2-cnzn-canzone-italiana/resolve/main/cnzn_sanremo.safetensors", "cnzn_sanremo.safetensors"),
    ],
    "yue2-chanson-francaise": [
        ("https://huggingface.co/becausereasons/yue2-chnsn-chanson-francaise/resolve/main/chnsn_montmartre.safetensors", "chnsn_montmartre.safetensors"),
    ],
    "yue2-epic-trailer": [
        ("https://huggingface.co/monsterovich/yue2-steps-from-hell/resolve/main/adapter-ar-195/lora.safetensors", "epic-ar.safetensors"),
        ("https://huggingface.co/monsterovich/yue2-steps-from-hell/resolve/main/adapter-nar-194/lora.safetensors", "epic-nar.safetensors"),
    ],
    "yue2-industrial-rock": [
        ("https://huggingface.co/monsterovich/yue2-industrial-rock-lora/resolve/main/adapter-ar-179/lora.safetensors", "industrial-ar.safetensors"),
        ("https://huggingface.co/monsterovich/yue2-industrial-rock-lora/resolve/main/adapter-nar-179-v2/lora.safetensors", "industrial-nar.safetensors"),
    ],
}


def setup_adapters(work_dir: Path) -> Path:
    """Configures adapters directory and restores/downloads catalogue LoRA styles."""
    adapters_dir = work_dir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    # 1. Restore from Kaggle input datasets if available (instant, 0s)
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for dataset_dir in input_root.iterdir():
            if not dataset_dir.is_dir():
                continue
            cand = dataset_dir / "adapters"
            if cand.exists() and cand.is_dir():
                for sub in cand.iterdir():
                    if sub.is_dir():
                        target_sub = adapters_dir / sub.name
                        target_sub.mkdir(parents=True, exist_ok=True)
                        for f in sub.glob("*.safetensors"):
                            target_file = target_sub / f.name
                            if not target_file.exists():
                                try:
                                    target_file.symlink_to(f)
                                except Exception:
                                    shutil.copy2(f, target_file)
                        meta_file = sub / "adapter.json"
                        if meta_file.exists():
                            try:
                                shutil.copy2(meta_file, target_sub / "adapter.json")
                            except Exception:
                                pass
                        logger.info("Restored adapter from dataset: %s", sub.name)

    # 2. Download primary 'yue2-instrumental' LoRA if not already present
    inst_dir = adapters_dir / "yue2-instrumental"
    inst_file = inst_dir / "ar_lora_inst_v3abc.bf16.safetensors"
    if not (inst_file.exists() and inst_file.stat().st_size > 10_000_000):
        inst_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = inst_dir / "ar_lora_inst.tmp"
        url = "https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras/resolve/main/ar_lora_inst_v3abc.bf16.safetensors"
        logger.info("Downloading primary 'yue2-instrumental' LoRA adapter...")
        try:
            cmd = ["curl", "-fSL", "--connect-timeout", "15", "--max-time", "120", "-o", str(tmp_file), url]
            subprocess.run(cmd, check=True, timeout=125)
            if tmp_file.exists() and tmp_file.stat().st_size > 10_000_000:
                tmp_file.replace(inst_file)
                meta = inst_dir / "adapter.json"
                meta.write_text(json.dumps({
                    "id": "yue2-instrumental",
                    "engine": "yue2-cpp",
                    "kind": "composition",
                    "name": {"en": "Instrumental", "ru": "Инструментал"},
                    "scales": {"ar": 1.0},
                    "slots": ["ar"]
                }))
                logger.info("✓ Ready: yue2-instrumental LoRA (%d MB)", inst_file.stat().st_size // (1024 * 1024))
        except Exception as exc:
            logger.warning("Could not download yue2-instrumental (will continue without it): %s", exc)
        finally:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)

    return adapters_dir


def launch_engine(
    executable: Path,
    models: dict[str, Path],
    port: int = YUE_BACKEND_PORT,
    gpu_id: int = 0,
    adapters_dir: Path | None = None,
) -> subprocess.Popen[str]:
    """Launches yue-server process."""
    cmd = [
        str(executable),
        "--model", str(models["backbone"]),
        "--vae", str(models["vae"]),
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    valid_adapters = []
    if adapters_dir and adapters_dir.exists():
        for sub in list(adapters_dir.iterdir()):
            if not sub.is_dir():
                continue
            # Remove any zero-byte or temporary files
            for f in list(sub.iterdir()):
                if f.name.endswith(".tmp") or (f.is_file() and f.stat().st_size < 1_000_000):
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass
            safetensors = [f for f in sub.glob("*.safetensors") if f.stat().st_size > 1_000_000]
            if safetensors:
                meta = sub / "adapter.json"
                if not meta.exists():
                    try:
                        meta.write_text(json.dumps({"id": sub.name, "engine": "yue2-cpp", "name": {"en": sub.name}}))
                    except Exception:
                        pass
                valid_adapters.append(sub.name)
            else:
                shutil.rmtree(sub, ignore_errors=True)

    if valid_adapters:
        cmd.extend(["--adapters", str(adapters_dir)])
        logger.info("Configured LoRA adapters for GPU %d (%s): %s", gpu_id, ", ".join(valid_adapters), adapters_dir)

    logger.info("Starting yue-server on GPU %d: %s", gpu_id, " ".join(cmd))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    work_dir = executable.parent
    cuda_paths = [
        str(work_dir),
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(cuda_paths) + ":" + env.get("LD_LIBRARY_PATH", "")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    return proc


def wait_for_engine_ready(proc: subprocess.Popen[str], port: int, timeout: int = 300) -> bool:
    """Waits for yue-server to load models and pass /health."""
    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    logger.info("Waiting for yue-server on port %d to answer /health (up to %ds)...", port, timeout)
    while time.time() - start < timeout:
        if proc.poll() is not None:
            logger.error("yue-server process exited unexpectedly with code %d!", proc.returncode)
            return False
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status in (200, 204):
                    logger.info("[OK] yue-server is ONLINE and READY on port %d! (took %.1fs)", port, time.time() - start)
                    return True
        except Exception:
            time.sleep(2)
    logger.warning("yue-server startup timeout exceeded (%ds).", timeout)
    return False


NGROK_AUTHTOKEN: Final[str] = "20E8KAqQ4AwqWWDD7r6qjDjYhjv_6qdTdqy1QahUD3aXV6yg3"


def setup_ngrok_tunnel(port: int, authtoken: str = NGROK_AUTHTOKEN) -> tuple[subprocess.Popen[str] | None, str]:
    """Installs/configures pyngrok and opens an authenticated public HTTPS tunnel."""
    try:
        import pyngrok
    except ImportError:
        logger.info("Installing pyngrok for authenticated high-speed tunnel...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"], check=False)

    from pyngrok import conf, ngrok

    conf.get_default().auth_token = authtoken
    try:
        ngrok.set_auth_token(authtoken)
    except Exception:
        pass
    try:
        ngrok.kill()
    except Exception:
        pass

    tunnel = ngrok.connect(port, proto="http")
    url = str(tunnel.public_url).strip()
    if url.startswith("http://"):
        url = "https://" + url[7:]
    logger.info("✓ ngrok authenticated tunnel online: %s", url)
    return None, url


def setup_primary_tunnel(port: int, work_dir: Path) -> tuple[subprocess.Popen[str] | None, str]:
    """Establishes high-speed Cloudflare Quick Tunnel without external credentials."""
    return setup_cloudflare_tunnel(port, work_dir)


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
        "--url", f"http://127.0.0.1:{port}",
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

    def drain_tunnel(p: subprocess.Popen[str]) -> None:
        while True:
            if p.stdout:
                line = p.stdout.readline()
                if not line and p.poll() is not None:
                    break
            time.sleep(0.02)

    threading.Thread(target=drain_tunnel, args=(proc,), daemon=True).start()
    return proc, public_url


def notify_ready(tunnel_url: str, username: str = "jungterdai") -> None:
    """Publishes tunnel URL across redundant discovery channels for YuE2 Studio."""
    h = hashlib.sha256(username.strip().lower().encode("utf-8")).hexdigest()[:16]
    topic = f"yue2_studio_{h}"

    for channel, u in [
        ("ntfy.sh", f"https://ntfy.sh/{topic}"),
        ("ntfy.envs.net", f"https://ntfy.envs.net/{topic}"),
    ]:
        try:
            req = urllib.request.Request(
                u,
                data=tunnel_url.encode("utf-8"),
                headers={
                    "Title": "YuE2 Studio Ready",
                    "Tags": "musical_note,guitar",
                    "Priority": "high",
                },
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("[*] Published to %s: %s", channel, topic)
        except Exception as exc:
            logger.warning("Could not publish to %s: %s", channel, exc)

    try:
        form = urllib.parse.urlencode({"content": tunnel_url}).encode("utf-8")
        req = urllib.request.Request(f"https://cl1p.net/{topic}", data=form)
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                logger.info("[*] Published to cl1p.net: %s", topic)
    except Exception as exc:
        logger.warning("Could not publish to cl1p.net: %s", exc)


def check_remote_shutdown_signal(username: str = "jungterdai") -> bool:
    """Checks for a shutdown command from YuE2 Studio application."""
    h = hashlib.sha256(username.strip().lower().encode("utf-8")).hexdigest()[:16]
    topic = f"yue2_studio_{h}_control"
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}/json?since=60s&poll=1",
            headers={"User-Agent": "YuE2-Runner"},
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
            for line in text.splitlines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "SHUTDOWN" in data.get("message", ""):
                            return True
                    except Exception:
                        if "SHUTDOWN" in line:
                            return True
    except Exception:
        pass
    return False


ALLOWED_PROXIED_PREFIXES: Final[tuple[str, ...]] = (
    "/synth",
    "/job",
    "/health",
    "/props",
    "/step",
)


def midi_to_abc_note(midi_num: int) -> str:
    """Converts a MIDI note number (0-127) to standard ABC notation note token."""
    names = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"]
    octave = (midi_num // 12) - 1
    name = names[midi_num % 12]
    if octave < 4:
        return name + ("," * (4 - octave))
    elif octave == 4:
        return name
    elif octave == 5:
        return name.lower()
    else:
        return name.lower() + ("'" * (octave - 5))


def transcribe_audio_to_abc(audio_bytes: bytes, melody_only: bool = True) -> str:
    """Extracts melody pitch contour and beats from audio bytes and generates 2-voice ABC score."""
    import tempfile
    import numpy as np

    bpm = 120
    notes_sequence: list[str] = []

    try:
        import librosa

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = Path(f.name)

        try:
            y, sr = librosa.load(str(tmp_path), sr=16000, duration=180.0)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            detected_bpm = int(np.round(np.atleast_1d(tempo)[0]))
            if 60 <= detected_bpm <= 200:
                bpm = detected_bpm

            # Extract fundamental frequency using probabilistic YIN
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
                frame_length=2048,
            )

            # Sample pitch at ~8th note intervals
            hop_time = 512 / sr
            sec_per_eighth = (60.0 / bpm) / 2.0
            step_frames = max(1, int(sec_per_eighth / hop_time))

            for i in range(0, len(f0), step_frames):
                if voiced_flag[i] and not np.isnan(f0[i]) and f0[i] > 50:
                    midi_val = int(np.round(12 * np.log2(f0[i] / 440.0) + 69))
                    midi_val = max(36, min(96, midi_val))
                    notes_sequence.append(midi_to_abc_note(midi_val) + "2")
                else:
                    notes_sequence.append("z2")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    except Exception as exc:
        logger.warning("librosa audio transcription fallback: %s", exc)

    if not notes_sequence:
        # Fallback melody contour
        default_midis = [60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 64, 62, 60]
        notes_sequence = [midi_to_abc_note(m) + "2" for m in default_midis]

    # Organize notes into 4/4 measures (each bar = 4 eighth pairs = 4 items of length 2)
    measures: list[str] = []
    chunk_size = 4
    for i in range(0, len(notes_sequence), chunk_size):
        chunk = notes_sequence[i : i + chunk_size]
        while len(chunk) < chunk_size:
            chunk.append("z2")
        measures.append(" ".join(chunk))

    sections = [
        ("[Intro]", 2),
        ("[Verse 1]", 4),
        ("[Chorus]", 4),
        ("[Verse 2]", 4),
        ("[Chorus]", 4),
        ("[Outro]", 2),
    ]

    lines = [
        "X:1",
        "T:Cover Melody",
        "M:4/4",
        "L:1/8",
        f"Q:1/4={bpm}",
        "K:C",
        'V:Vocal clef=treble name="Vocal Melody" snm="Vocal"',
        'V:Ins clef=treble name="Ins Melody" snm="Inst."',
    ]

    m_idx = 0
    for sec_tag, num_bars in sections:
        lines.append(f"[V:Vocal] {sec_tag}")
        sec_bars = []
        for _ in range(num_bars):
            if m_idx < len(measures):
                sec_bars.append(measures[m_idx])
                m_idx += 1
            else:
                sec_bars.append("C2 E2 G2 c2")
        lines.append("| " + " | ".join(sec_bars) + " |")

    return "\n".join(lines)


COMPLETED_JOBS: dict[str, str] = {}
LAST_ACTIVITY_TIME: float = time.time()


def extract_multipart_parts(content_type: str, body: bytes) -> list[tuple[str, bytes]]:
    """Extracts (content_type, part_bytes) from a multipart/mixed HTTP body safely without corrupting binary data."""
    if not content_type.lower().startswith("multipart/"):
        return [(content_type, body)] if body else []

    boundary = ""
    for token in content_type.split(";"):
        token = token.strip()
        if token.lower().startswith("boundary="):
            boundary = token.split("=", 1)[1].strip("\"'")
            break
    if not boundary:
        return []

    marker = b"--" + boundary.encode("utf-8")
    if not body.startswith(marker):
        return []

    cursor = 0
    parts: list[tuple[str, bytes]] = []
    while cursor < len(body):
        if not body[cursor:].startswith(marker):
            break
        cursor += len(marker)
        if body[cursor:].startswith(b"--\r\n") or body[cursor:] == b"--":
            break
        if not body[cursor:].startswith(b"\r\n"):
            break
        cursor += 2
        hdr_end = body[cursor:].find(b"\r\n\r\n")
        if hdr_end == -1:
            break
        hdr_end += cursor
        hdr_str = body[cursor:hdr_end].decode("utf-8", errors="ignore")
        ctype = "application/octet-stream"
        for line in hdr_str.split("\r\n"):
            if line.lower().startswith("content-type:"):
                ctype = line.split(":", 1)[1].strip()
        cstart = hdr_end + 4
        next_b = body[cstart:].find(b"\r\n" + marker)
        if next_b == -1:
            break
        next_b += cstart
        part_body = body[cstart:next_b]
        parts.append((ctype, part_body))
        cursor = next_b + 2
    return parts


class YuE2ServerHandler(http.server.BaseHTTPRequestHandler):
    """HTTP proxy for YuE2 engine: adds /shutdown, /transcribe and transparently forwards valid API calls."""

    backend_port: int = YUE_BACKEND_PORT

    def _mark_activity(self) -> None:
        global LAST_ACTIVITY_TIME
        # Health probes, props, and log queries do not reset the auto-idle shutdown timer
        if not (self.path.startswith("/health") or self.path.startswith("/props") or self.path.startswith("/logs")):
            LAST_ACTIVITY_TIME = time.time()

    def do_GET(self) -> None:
        self._mark_activity()
        if self.path == "/shutdown":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Shutting down YuE2 instance...")
            logger.info("Shutdown requested via /shutdown.")
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0))).start()
            return

        if self.path.startswith("/logs"):
            uptime_sec = int(time.time() - START_TIME)
            idle_sec = int(time.time() - LAST_ACTIVITY_TIME)
            idle_left = max(0, 600 - idle_sec)
            header = [
                "=" * 60,
                "YuE2 Studio Live Server Monitor (Kaggle)",
                f"Uptime: {uptime_sec // 60}m {uptime_sec % 60}s | Inactive: {idle_sec // 60}m {idle_sec % 60}s / 10m00s",
                f"Auto-shutdown in: {idle_left // 60}m {idle_left % 60}s (quits after 10m idle)",
                f"Mode: {'Dual-GPU (2x Tesla T4)' if DUAL_GPU_ENABLED else 'Single-GPU'}",
                f"Active dual jobs: {len(ACTIVE_DUAL_JOBS)}",
                "=" * 60,
            ]
            content = "\n".join(header + list(LOG_BUFFER))
            resp_data = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(resp_data)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(resp_data)
            return

        if self.path.startswith("/props"):
            desc = "YuE2 Dedicated (Kaggle 2x Tesla T4 GPU, Dual-Variation)" if DUAL_GPU_ENABLED else "YuE2 Dedicated (Kaggle Tesla T4 GPU)"
            resp_data = json.dumps({
                "version": desc,
                "gpus": 2 if DUAL_GPU_ENABLED else 1,
                "dual_gpu": DUAL_GPU_ENABLED,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_data)))
            self.end_headers()
            self.wfile.write(resp_data)
            return

        if self.path.startswith("/job"):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            job_id = qs.get("id", [""])[0]

            if job_id in ACTIVE_DUAL_JOBS:
                self._handle_dual_job(job_id, qs)
                return

            if job_id in COMPLETED_JOBS:
                abc = COMPLETED_JOBS[job_id]
                if qs.get("result", [""])[0] == "1":
                    resp_data = json.dumps({"abc": abc}).encode("utf-8")
                else:
                    resp_data = json.dumps({"status": "done", "abc": abc}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_data)))
                self.end_headers()
                self.wfile.write(resp_data)
                return

        if any(self.path.startswith(prefix) for prefix in ALLOWED_PROXIED_PREFIXES):
            self._proxy_to_backend("GET")
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Unknown route: {self.path}"}).encode("utf-8"))

    def do_POST(self) -> None:
        self._mark_activity()
        if self.path == "/shutdown":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Shutting down YuE2 instance...")
            logger.info("Shutdown requested via /shutdown.")
            threading.Thread(target=lambda: (time.sleep(0.5), os._exit(0))).start()
            return

        if self.path.startswith("/synth") and DUAL_GPU_ENABLED:
            self._handle_dual_synth()
            return

        if self.path.startswith("/transcribe"):
            self._handle_transcribe()
            return

        if any(self.path.startswith(prefix) for prefix in ALLOWED_PROXIED_PREFIXES):
            self._proxy_to_backend("POST")
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Unknown route: {self.path}"}).encode("utf-8"))

    def do_HEAD(self) -> None:
        if any(self.path.startswith(prefix) for prefix in ALLOWED_PROXIED_PREFIXES):
            self._proxy_to_backend("HEAD")
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_transcribe(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # First attempt: proxy directly to native yue-server (SheetSage2)
        target = f"http://127.0.0.1:{self.backend_port}/transcribe"
        req_headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
        try:
            req = urllib.request.Request(target, data=body, headers=req_headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status in (200, 201, 202):
                    resp_data = resp.read()
                    self.send_response(resp.status)
                    for header, value in resp.headers.items():
                        if header.lower() not in ("transfer-encoding", "content-length"):
                            self.send_header(header, value)
                    self.send_header("Content-Length", str(len(resp_data)))
                    self.end_headers()
                    self.wfile.write(resp_data)
                    logger.info("Proxied /transcribe successfully to yue-server (SheetSage2).")
                    return
        except Exception as p_exc:
            logger.warning("yue-server native transcribe unavailable (%s), using fallback...", p_exc)

        # Fallback to Python transcription
        audio_bytes = body
        melody_only = True
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type and b"\r\n\r\n" in body:
            parts = body.split(b"\r\n\r\n")
            if len(parts) >= 2:
                tail = parts[1]
                if b"\r\n--" in tail:
                    audio_bytes = tail.split(b"\r\n--")[0]
                else:
                    audio_bytes = tail

        logger.info("Received /transcribe request (%d audio bytes). Extracting ABC score...", len(audio_bytes))
        abc_score = transcribe_audio_to_abc(audio_bytes, melody_only=melody_only)

        job_id = f"transcribe_{int(time.time())}"
        COMPLETED_JOBS[job_id] = abc_score

        resp_obj = {
            "id": job_id,
            "status": "done",
            "abc": abc_score,
        }
        resp_data = json.dumps(resp_obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_data)))
        self.end_headers()
        self.wfile.write(resp_data)
        logger.info("Transcribed audio successfully into ABC score.")
    def _handle_dual_synth(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            data = json.loads(raw_body.decode("utf-8"))
        except Exception:
            data = {}

        seed0 = data.get("seed", int(time.time() * 1000) % 1_000_000_000)
        seed1 = (seed0 + 10007) % 2_000_000_000

        data0 = dict(data)
        data0["seed"] = seed0
        data0["lm_seed"] = seed0

        data1 = dict(data)
        data1["seed"] = seed1
        data1["lm_seed"] = seed1

        body0 = json.dumps(data0).encode("utf-8")
        body1 = json.dumps(data1).encode("utf-8")

        headers_base = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length")}
        headers_base["Content-Type"] = "application/json"

        h0 = dict(headers_base)
        h0["Content-Length"] = str(len(body0))

        h1 = dict(headers_base)
        h1["Content-Length"] = str(len(body1))

        # Dispatch Worker 0 (GPU 0)
        id0 = ""
        try:
            req0 = urllib.request.Request(f"http://127.0.0.1:{YUE_BACKEND_PORT}/synth", data=body0, headers=h0, method="POST")
            with urllib.request.urlopen(req0, timeout=120) as resp0:
                resp0_data = json.loads(resp0.read().decode("utf-8"))
                id0 = str(resp0_data.get("id", resp0_data.get("job_id", "")))
        except Exception as exc:
            logger.error("Failed to submit synthesis to Worker 0: %s", exc)

        # Dispatch Worker 1 (GPU 1)
        id1 = ""
        try:
            req1 = urllib.request.Request(f"http://127.0.0.1:{YUE_BACKEND_PORT_2}/synth", data=body1, headers=h1, method="POST")
            with urllib.request.urlopen(req1, timeout=120) as resp1:
                resp1_data = json.loads(resp1.read().decode("utf-8"))
                id1 = str(resp1_data.get("id", resp1_data.get("job_id", "")))
        except Exception as exc:
            logger.error("Failed to submit synthesis to Worker 1: %s", exc)

        if not id0 and not id1:
            err_msg = json.dumps({"error": "Failed to submit synthesis to both GPU workers"}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_msg)))
            self.end_headers()
            self.wfile.write(err_msg)
            return

        if id0 and id1:
            composite_id = f"{id0}__{id1}"
            ACTIVE_DUAL_JOBS[composite_id] = (id0, id1)
            resp_obj = {
                "id": composite_id,
                "job_id": composite_id,
                "status": "running",
                "variations": 2,
            }
        else:
            single_id = id0 or id1
            resp_obj = {
                "id": single_id,
                "job_id": single_id,
                "status": "running",
                "variations": 1,
            }

        resp_data = json.dumps(resp_obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_data)))
        self.end_headers()
        self.wfile.write(resp_data)
        logger.info("Dispatched dual synthesis to 2x Tesla T4 GPUs (Composite Job: %s)", resp_obj["id"])

    def _handle_dual_job(self, composite_id: str, qs: dict[str, list[str]]) -> None:
        (id0, id1) = ACTIVE_DUAL_JOBS[composite_id]
        result_param = qs.get("result", [""])[0]

        if result_param == "2":
            target = f"http://127.0.0.1:{YUE_BACKEND_PORT_2}/job?id={id1}&result=1"
            self._proxy_to_backend("GET", target=target)
            return
        elif result_param == "1":
            # Fetch Worker 0
            body0, ct0, status0 = b"", "", 500
            try:
                req0 = urllib.request.Request(f"http://127.0.0.1:{YUE_BACKEND_PORT}/job?id={id0}&result=1", headers={"User-Agent": "DualMerger"})
                with urllib.request.urlopen(req0, timeout=120) as r0:
                    status0 = r0.status
                    ct0 = r0.headers.get("Content-Type", "")
                    body0 = r0.read()
            except Exception as exc:
                logger.warning("Could not fetch result from Worker 0: %s", exc)

            # Fetch Worker 1
            body1, ct1, status1 = b"", "", 500
            try:
                req1 = urllib.request.Request(f"http://127.0.0.1:{YUE_BACKEND_PORT_2}/job?id={id1}&result=1", headers={"User-Agent": "DualMerger"})
                with urllib.request.urlopen(req1, timeout=120) as r1:
                    status1 = r1.status
                    ct1 = r1.headers.get("Content-Type", "")
                    body1 = r1.read()
            except Exception as exc:
                logger.warning("Could not fetch result from Worker 1: %s", exc)

            # Extract parts and merge so desktop app receives all variations at once!
            parts0 = extract_multipart_parts(ct0, body0) if status0 == 200 else []
            parts1 = extract_multipart_parts(ct1, body1) if status1 == 200 else []

            if parts0 and parts1:
                boundary = "yue2-dual-composite-boundary"
                combined = b""
                for p_type, p_data in (parts0 + parts1):
                    combined += f"--{boundary}\r\nContent-Type: {p_type}\r\n\r\n".encode("utf-8")
                    combined += p_data
                    combined += b"\r\n"
                combined += f"--{boundary}--\r\n".encode("utf-8")

                self.send_response(200)
                self.send_header("Content-Type", f"multipart/mixed; boundary={boundary}")
                self.send_header("Content-Length", str(len(combined)))
                self.end_headers()
                self.wfile.write(combined)
                logger.info("✓ Combined both variations into single multipart/mixed response for %s (%d bytes).", composite_id, len(combined))
                return
            elif parts0:
                self.send_response(200)
                self.send_header("Content-Type", ct0)
                self.send_header("Content-Length", str(len(body0)))
                self.end_headers()
                self.wfile.write(body0)
                return
            elif parts1:
                self.send_response(200)
                self.send_header("Content-Type", ct1)
                self.send_header("Content-Length", str(len(body1)))
                self.end_headers()
                self.wfile.write(body1)
                return
            else:
                self.send_response(502)
                self.end_headers()
                self.wfile.write(b'{"error": "Failed to retrieve results from workers"}')
                return

        status0 = "running"
        status1 = "running"
        try:
            req0 = urllib.request.Request(f"http://127.0.0.1:{YUE_BACKEND_PORT}/job?id={id0}", headers={"User-Agent": "DualProxy"})
            with urllib.request.urlopen(req0, timeout=10) as r0:
                d0 = json.loads(r0.read().decode("utf-8"))
                status0 = d0.get("status", "running")
        except Exception:
            pass

        try:
            req1 = urllib.request.Request(f"http://127.0.0.1:{YUE_BACKEND_PORT_2}/job?id={id1}", headers={"User-Agent": "DualProxy"})
            with urllib.request.urlopen(req1, timeout=10) as r1:
                d1 = json.loads(r1.read().decode("utf-8"))
                status1 = d1.get("status", "running")
        except Exception:
            pass

        if status0 == "done" and status1 == "done":
            combined_status = "done"
        elif status0 in ("failed", "error") and status1 in ("failed", "error"):
            combined_status = "failed"
        elif status0 == "done" and status1 in ("failed", "error"):
            combined_status = "done"
        elif status1 == "done" and status0 in ("failed", "error"):
            combined_status = "done"
        else:
            combined_status = "running"

        resp_obj = {
            "id": composite_id,
            "status": combined_status,
            "variations": 2,
            "worker0": status0,
            "worker1": status1,
        }
        resp_data = json.dumps(resp_obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_data)))
        self.end_headers()
        self.wfile.write(resp_data)

    def _proxy_to_backend(self, method: str, target: str | None = None) -> None:
        if target is None:
            target = f"http://127.0.0.1:{self.backend_port}{self.path}"
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        req_headers: dict[str, str] = {}
        for header, value in self.headers.items():
            if header.lower() not in ("host", "content-length"):
                req_headers[header] = value

        req = urllib.request.Request(target, data=body, headers=req_headers, method=method)
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
            for header, value in exc.headers.items():
                if header.lower() not in ("transfer-encoding", "content-length"):
                    self.send_header(header, value)
            err_data = exc.read()
            self.send_header("Content-Length", str(len(err_data)))
            self.end_headers()
            self.wfile.write(err_data)
        except Exception as exc:
            logger.error("yue-server proxy error: %s", exc)
            err_msg = json.dumps({"error": f"yue-server proxy error: {exc}"}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err_msg)))
            self.end_headers()
            self.wfile.write(err_msg)

    def log_message(self, fmt: str, *args: object) -> None:
        pass


class ReusableServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True


def main() -> None:
    banner = "=" * 70 + "\n" + "YuE2 Studio Dedicated Server -- Kaggle Tesla T4\n" + "=" * 70
    print(banner, flush=True)

    work_dir, gpu_count = detect_environment()
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.call(["killall", "-9", "yue-server", "cloudflared"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    engine_bin = restore_yue_engine(work_dir)
    models = resolve_yue_models(work_dir)
    adapters_dir = setup_adapters(work_dir)

    global DUAL_GPU_ENABLED
    yue_proc: subprocess.Popen[str] | None = None
    yue_proc_2: subprocess.Popen[str] | None = None

    def stream_engine_logs(proc: subprocess.Popen[str], tag: str = "GPU") -> None:
        while True:
            if proc.stdout:
                line = proc.stdout.readline()
                if line:
                    clean = line.rstrip()
                    if clean:
                        LOG_BUFFER.append(f"{time.strftime('%H:%M:%S')} [{tag}] {clean}")
                    sys.stdout.write(f"[{tag}] {line}")
                    sys.stdout.flush()
                elif proc.poll() is not None:
                    break
            time.sleep(0.01)

    if gpu_count >= 2:
        logger.info("[Dual-GPU Mode] Initializing 2x Tesla T4 GPUs...")
        proc0 = launch_engine(engine_bin, models, port=YUE_BACKEND_PORT, gpu_id=0, adapters_dir=adapters_dir)
        proc1 = launch_engine(engine_bin, models, port=YUE_BACKEND_PORT_2, gpu_id=1, adapters_dir=adapters_dir)

        threading.Thread(target=stream_engine_logs, args=(proc0, "GPU-0"), daemon=True).start()
        threading.Thread(target=stream_engine_logs, args=(proc1, "GPU-1"), daemon=True).start()

        ready0 = wait_for_engine_ready(proc0, YUE_BACKEND_PORT, timeout=300)
        ready1 = wait_for_engine_ready(proc1, YUE_BACKEND_PORT_2, timeout=300)

        if not (ready0 or ready1) and adapters_dir is not None:
            logger.warning("GPU workers failed with adapters. Retrying cleanly WITHOUT adapters...")
            proc0 = launch_engine(engine_bin, models, port=YUE_BACKEND_PORT, gpu_id=0, adapters_dir=None)
            proc1 = launch_engine(engine_bin, models, port=YUE_BACKEND_PORT_2, gpu_id=1, adapters_dir=None)
            threading.Thread(target=stream_engine_logs, args=(proc0, "GPU-0"), daemon=True).start()
            threading.Thread(target=stream_engine_logs, args=(proc1, "GPU-1"), daemon=True).start()
            ready0 = wait_for_engine_ready(proc0, YUE_BACKEND_PORT, timeout=300)
            ready1 = wait_for_engine_ready(proc1, YUE_BACKEND_PORT_2, timeout=300)

        if ready0 and ready1:
            DUAL_GPU_ENABLED = True
            yue_proc = proc0
            yue_proc_2 = proc1
            logger.info("✓ [Dual-GPU Mode] Both Tesla T4 GPU workers are ONLINE and ready for parallel generation!")
        elif ready0:
            logger.warning("GPU 1 worker failed to start; operating in single GPU mode on GPU 0.")
            DUAL_GPU_ENABLED = False
            yue_proc = proc0
        elif ready1:
            logger.warning("GPU 0 worker failed to start; operating in single GPU mode on GPU 1.")
            DUAL_GPU_ENABLED = False
            yue_proc = proc1
            YuE2ServerHandler.backend_port = YUE_BACKEND_PORT_2
        else:
            logger.error("Both GPU workers failed to start.")
            sys.exit(1)
    else:
        DUAL_GPU_ENABLED = False
        yue_proc = launch_engine(engine_bin, models, port=YUE_BACKEND_PORT, gpu_id=0, adapters_dir=adapters_dir)
        threading.Thread(target=stream_engine_logs, args=(yue_proc, "GPU-0"), daemon=True).start()
        ready = wait_for_engine_ready(yue_proc, YUE_BACKEND_PORT, timeout=300)
        if not ready and adapters_dir is not None:
            logger.warning("GPU worker failed with adapters. Retrying WITHOUT adapters...")
            yue_proc = launch_engine(engine_bin, models, port=YUE_BACKEND_PORT, gpu_id=0, adapters_dir=None)
            threading.Thread(target=stream_engine_logs, args=(yue_proc, "GPU-0"), daemon=True).start()
            ready = wait_for_engine_ready(yue_proc, YUE_BACKEND_PORT, timeout=300)
        if not ready:
            logger.error("Failed to start yue-server in time.")
            sys.exit(1)

    # Start HTTP proxy on DEFAULT_PORT (18087)
    httpd = ReusableServer(("0.0.0.0", DEFAULT_PORT), YuE2ServerHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    logger.info("YuE2 HTTP proxy server active on port %d (Dual GPU: %s).", DEFAULT_PORT, DUAL_GPU_ENABLED)

    tunnel_proc, tunnel_url = setup_primary_tunnel(DEFAULT_PORT, work_dir)
    logger.info("[+] YuE2 Studio Public URL: %s", tunnel_url)
    notify_ready(tunnel_url)

    global LAST_ACTIVITY_TIME
    LAST_ACTIVITY_TIME = time.time()
    last_heartbeat = time.time()
    last_signal_check = time.time()
    try:
        while True:
            # Check primary worker process
            if yue_proc is not None and yue_proc.poll() is not None:
                logger.warning("Primary yue-server exited! Auto-restarting engine...")
                yue_proc = launch_engine(engine_bin, models, port=YUE_BACKEND_PORT, gpu_id=0)
                threading.Thread(target=stream_engine_logs, args=(yue_proc, "GPU-0"), daemon=True).start()
                wait_for_engine_ready(yue_proc, YUE_BACKEND_PORT, timeout=300)

            # Check secondary worker process if dual-GPU active
            if DUAL_GPU_ENABLED and yue_proc_2 is not None and yue_proc_2.poll() is not None:
                logger.warning("Secondary yue-server exited! Auto-restarting engine on GPU 1...")
                yue_proc_2 = launch_engine(engine_bin, models, port=YUE_BACKEND_PORT_2, gpu_id=1)
                threading.Thread(target=stream_engine_logs, args=(yue_proc_2, "GPU-1"), daemon=True).start()
                wait_for_engine_ready(yue_proc_2, YUE_BACKEND_PORT_2, timeout=300)

            # Check tunnel process if subprocess-based
            if tunnel_proc is not None and tunnel_proc.poll() is not None:
                logger.warning("Tunnel exited! Re-establishing tunnel...")
                tunnel_proc, tunnel_url = setup_primary_tunnel(DEFAULT_PORT, work_dir)
                logger.info("[+] New Public URL: %s", tunnel_url)
                notify_ready(tunnel_url)

            # Periodic heartbeat (every 30s) to keep session active and publish current URL
            now = time.time()
            if now - last_heartbeat >= 30:
                last_heartbeat = now
                mode_str = "2x Tesla T4 (Dual-Variation)" if DUAL_GPU_ENABLED else "Tesla T4"
                logger.info("[Heartbeat] YuE2 Engine online on Kaggle %s. Tunnel: %s", mode_str, tunnel_url)
                notify_ready(tunnel_url)

            # Check remote shutdown signal from application (every 6 seconds)
            if now - last_signal_check >= 6:
                last_signal_check = now
                if check_remote_shutdown_signal():
                    logger.info("[Remote Shutdown] Получен сигнал выключения сервера от пользователя. Завершение работы...")
                    break

            # Auto-idle shutdown after 10 minutes (600s) of inactivity to protect GPU quota & prevent bans
            idle_duration = time.time() - LAST_ACTIVITY_TIME
            if idle_duration >= 600:
                logger.info(
                    "[Auto-Idle Shutdown] Сервер бездействовал %d мин. "
                    "Завершение работы для сохранения квоты GPU Kaggle.",
                    int(idle_duration // 60),
                )
                try:
                    notify_ready("YuE2 Server остановлен по неактивности (10 мин). Квота сохранена.")
                except Exception:
                    pass
                break

            time.sleep(2)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        try:
            if yue_proc is not None:
                yue_proc.terminate()
        except Exception:
            pass
        try:
            if yue_proc_2 is not None:
                yue_proc_2.terminate()
        except Exception:
            pass
        try:
            if tunnel_proc is not None:
                tunnel_proc.terminate()
        except Exception:
            pass
        logger.info("Session cleanly terminated.")
        os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        logger.critical("Fatal crash: %s\n%s", exc, tb)
        try:
            h = hashlib.sha256(b"jungterdai").hexdigest()[:16]
            req = urllib.request.Request(
                f"https://ntfy.envs.net/yue2_studio_{h}",
                data=f"CRASH: {exc}".encode("utf-8"),
                headers={"Title": "YuE2 Server Crash", "Priority": "urgent"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
        sys.exit(1)
