"""ACE-Step Studio Dedicated External Engine Runner for Kaggle (Dual Tesla T4).

Dedicated specifically to ACE-Step 1.5 Turbo DiT and MuScriptor Audio-to-MIDI.
- GPU 0: ace-server (ACE-Step 1.5 Turbo XL DiT + LM 1.7B + Qwen3 Embedding + VAE)
- GPU 1: MuScriptor (Audio-to-MIDI transcription)
Restores precompiled ace-server and models from /kaggle/input (0s download time).
Establishes Cloudflare tunnel and announces URL to ACE-Step Studio applications.
"""

from __future__ import annotations

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
import uuid
from pathlib import Path
from typing import Final

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger: logging.Logger = logging.getLogger("acestep-server")

DEFAULT_PORT: Final[int] = 18087
ACE_BACKEND_PORT: Final[int] = 18089
CLOUDFLARED_URL: Final[str] = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
)


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


def restore_acestep_engine(work_dir: Path) -> Path:
    """Restores pre-compiled ace-server binary and libraries from Kaggle input."""
    target_bin = work_dir / "ace-server"
    local_libs = list(work_dir.glob("lib*.so*"))

    if target_bin.exists() and local_libs:
        target_bin.chmod(0o755)
        logger.info("Cached ace-server found at %s", target_bin)
        return target_bin

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for candidate in kaggle_input.rglob("ace-server"):
            if candidate.is_file():
                cand_dir = candidate.parent
                cand_libs = list(cand_dir.glob("lib*.so*"))
                if cand_libs:
                    shutil.copy2(candidate, target_bin)
                    target_bin.chmod(0o755)
                    for lib in cand_libs:
                        shutil.copy2(lib, work_dir / lib.name)
                    logger.info("Restored ace-server from Kaggle Input: %s", candidate)
                    return target_bin

    raise FileNotFoundError("Could not find pre-compiled ace-server in /kaggle/input/acestep-models.")


def resolve_acestep_models(work_dir: Path) -> Path:
    """Symlinks required GGUF weights from /kaggle/input into models directory."""
    models_dir = work_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    kaggle_input = Path("/kaggle/input")

    required = [
        "acestep-v15-xl-turbo-Q8_0.gguf",
        "acestep-5Hz-lm-1.7B-Q8_0.gguf",
        "Qwen3-Embedding-0.6B-Q8_0.gguf",
        "vae-BF16.gguf",
    ]

    for filename in required:
        dest = models_dir / filename
        if dest.exists() and dest.stat().st_size > 1024 * 1024:
            continue
        matches = list(kaggle_input.rglob(filename)) if kaggle_input.exists() else []
        valid = [m for m in matches if m.is_file() and m.stat().st_size > 1024 * 1024]
        if valid:
            logger.info("Found '%s' in Kaggle dataset: %s", filename, valid[0])
            try:
                if dest.is_symlink() or dest.exists():
                    dest.unlink(missing_ok=True)
                dest.symlink_to(valid[0])
            except Exception:
                shutil.copy2(valid[0], dest)
        else:
            raise FileNotFoundError(f"Required ACE-Step model {filename} not found in Kaggle input.")

    return models_dir


def launch_acestep_engine(
    executable: Path,
    models_dir: Path,
    port: int = ACE_BACKEND_PORT,
    gpu_id: int = 0,
) -> subprocess.Popen[str]:
    """Launches ace-server process on designated GPU."""
    cmd = [
        str(executable),
        "--models", str(models_dir),
        "--host", "0.0.0.0",
        "--port", str(port),
        "--keep-loaded",
    ]

    logger.info("Starting ace-server on GPU %d: %s", gpu_id, " ".join(cmd))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["GGML_BACKEND"] = "CUDA"

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


def wait_for_engine_ready(port: int, timeout: int = 60) -> bool:
    """Waits for ace-server to load models and answer /health."""
    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    logger.info("Waiting for ace-server to answer /health...")
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status in (200, 204):
                    logger.info("✓ ace-server is ONLINE and READY on port %d!", port)
                    return True
        except Exception:
            time.sleep(1)
    logger.warning("ace-server startup timeout exceeded.")
    return False


TRANSCRIBE_JOBS: dict[str, dict[str, object]] = {}


def ensure_muscriptor_model(work_dir: Path) -> Path:
    """Ensures MuScriptor weights are resolved from /kaggle/input."""
    target_dir = work_dir / "models" / "muscriptor-large"
    target_dir.mkdir(parents=True, exist_ok=True)
    safetensors = target_dir / "model.safetensors"
    config_json = target_dir / "config.json"

    if safetensors.exists() and safetensors.stat().st_size > 1024 * 1024:
        return safetensors

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates = list(kaggle_input.rglob("model.safetensors"))
        valid = [c for c in candidates if c.is_file() and c.stat().st_size > 100 * 1024 * 1024]
        if valid:
            found_st = valid[0]
            found_cfg = found_st.parent / "config.json"
            logger.info("Found MuScriptor weights in Kaggle dataset: %s", found_st)
            try:
                safetensors.symlink_to(found_st)
                if found_cfg.exists():
                    config_json.symlink_to(found_cfg)
                return safetensors
            except Exception:
                return found_st

    logger.info("MuScriptor weights not found in dataset. Will download on demand if needed.")
    return safetensors


class AceStudioGateway(http.server.BaseHTTPRequestHandler):
    """Unified HTTP gateway forwarding requests to ace-server and MuScriptor."""

    backend_port: int = ACE_BACKEND_PORT
    work_dir: Path = Path("/kaggle/working")
    transcribe_gpu_id: int = 1

    def do_GET(self) -> None:
        if self.path in ("/health", "/healthz"):
            self._proxy_to_backend("GET")
            return
        if self.path == "/props":
            self._proxy_to_backend("GET")
            return
        if self.path.startswith("/job"):
            self._proxy_to_backend("GET")
            return
        if self.path.startswith("/transcribe"):
            self._handle_transcribe_get()
            return
        self._proxy_to_backend("GET")

    def do_POST(self) -> None:
        if self.path == "/synth":
            self._proxy_to_backend("POST")
            return
        if self.path.startswith("/transcribe"):
            self._handle_transcribe_post()
            return
        self._proxy_to_backend("POST")

    def do_HEAD(self) -> None:
        self._proxy_to_backend("HEAD")

    def _proxy_to_backend(self, method: str) -> None:
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
            self.end_headers()
            self.wfile.write(exc.read())
        except Exception as exc:
            logger.error("ace-server proxy error: %s", exc)
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(exc).encode())

    def _handle_transcribe_get(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        job_id = qs.get("id", [""])[0].strip()
        job = TRANSCRIBE_JOBS.get(job_id)
        if not job:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Job not found"}')
            return

        if "result" in qs:
            out_file = job.get("output")
            if isinstance(out_file, Path) and out_file.exists():
                data = out_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "audio/midi")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(404)
            self.end_headers()
            return

        resp_data = json.dumps({"status": job.get("status", "running")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_data)))
        self.end_headers()
        self.wfile.write(resp_data)

    def _handle_transcribe_post(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        audio_data = self.rfile.read(content_length)
        job_id = f"transcribe_{uuid.uuid4().hex[:12]}"
        tmp_in = Path(f"/tmp/in_{job_id}.mp3")
        tmp_out = Path(f"/tmp/out_{job_id}.mid")
        tmp_in.write_bytes(audio_data)

        def worker() -> None:
            try:
                if shutil.which("muscriptor") is None:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "muscriptor"])
                model_path = ensure_muscriptor_model(self.work_dir)

                # Convert to WAV for robust decoding
                tmp_wav = tmp_in.with_suffix(".wav")
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(tmp_in), "-ar", "44100", "-ac", "2", str(tmp_wav)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
                    )
                except Exception:
                    pass
                input_file = tmp_wav if tmp_wav.exists() and tmp_wav.stat().st_size > 0 else tmp_in

                cmd = [
                    "muscriptor", "transcribe", str(input_file), "-o", str(tmp_out),
                    "--model", str(model_path), "--device", "cuda", "--dtype", "float16"
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(self.transcribe_gpu_id)

                res = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if res.returncode == 0 and tmp_out.exists():
                    TRANSCRIBE_JOBS[job_id]["status"] = "finished"
                    TRANSCRIBE_JOBS[job_id]["output"] = tmp_out
                else:
                    TRANSCRIBE_JOBS[job_id]["status"] = "failed"
                    TRANSCRIBE_JOBS[job_id]["error"] = res.stderr
            except Exception as e:
                TRANSCRIBE_JOBS[job_id]["status"] = "failed"
                TRANSCRIBE_JOBS[job_id]["error"] = str(e)

        TRANSCRIBE_JOBS[job_id] = {"status": "running"}
        threading.Thread(target=worker, daemon=True).start()

        resp_data = json.dumps({"job_id": job_id, "status": "running"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_data)))
        self.end_headers()
        self.wfile.write(resp_data)

    def log_message(self, format: str, *args: object) -> None:
        if args and str(args[0]).startswith("GET /health"):
            return
        logger.info("%s - %s", self.address_string(), format % args)


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

    return proc, public_url


def notify_ready(tunnel_url: str, username: str = "alexeyklochkov") -> None:
    """Publishes tunnel URL across redundant discovery channels for ACE-Step Studio."""
    h = hashlib.sha256(username.strip().lower().encode("utf-8")).hexdigest()[:16]
    topic = f"acestep_studio_{h}"

    for channel, u in [
        ("ntfy.envs.net", f"https://ntfy.envs.net/{topic}"),
        ("ntfy.sh", f"https://ntfy.sh/{topic}"),
    ]:
        try:
            req = urllib.request.Request(
                u,
                data=tunnel_url.encode("utf-8"),
                headers={
                    "Title": "ACE-Step Studio Ready",
                    "Tags": "zap,rocket",
                    "Priority": "high",
                },
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("📡 Published to %s: %s", channel, topic)
        except Exception as exc:
            logger.warning("Could not publish to %s: %s", channel, exc)

    try:
        form = urllib.parse.urlencode({"content": tunnel_url}).encode("utf-8")
        req = urllib.request.Request(f"https://cl1p.net/{topic}", data=form)
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                logger.info("📡 Published to cl1p.net: %s", topic)
    except Exception as exc:
        logger.warning("Could not publish to cl1p.net: %s", exc)


def main() -> None:
    banner = "=" * 70 + "\n" + "⚡ ACE-Step 1.5 Turbo Dedicated Server — Kaggle 2x Tesla T4\n" + "=" * 70
    print(banner, flush=True)

    work_dir, gpu_count = detect_environment()
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.call(["killall", "-9", "ace-server", "cloudflared"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    engine_bin = restore_acestep_engine(work_dir)
    models_dir = resolve_acestep_models(work_dir)

    ace_proc = launch_acestep_engine(engine_bin, models_dir, port=ACE_BACKEND_PORT, gpu_id=0)
    ready = wait_for_engine_ready(ACE_BACKEND_PORT, timeout=60)
    if not ready:
        logger.error("Failed to start ace-server in time.")

    # Start gateway on primary port
    AceStudioGateway.backend_port = ACE_BACKEND_PORT
    AceStudioGateway.work_dir = work_dir
    AceStudioGateway.transcribe_gpu_id = 1 if gpu_count >= 2 else 0

    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", DEFAULT_PORT), AceStudioGateway)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    tunnel_proc, tunnel_url = setup_cloudflare_tunnel(DEFAULT_PORT, work_dir)
    logger.info("🎉 ACE-Step Studio Public URL: %s", tunnel_url)
    notify_ready(tunnel_url)

    try:
        while True:
            if ace_proc.stdout:
                line = ace_proc.stdout.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            time.sleep(0.01)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        ace_proc.terminate()
        tunnel_proc.terminate()
        httpd.server_close()


if __name__ == "__main__":
    main()
