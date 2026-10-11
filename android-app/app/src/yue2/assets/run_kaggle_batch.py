"""YuE2 Studio Batch Music Synthesizer (Zero-Tunnel Architecture).

Runs a single-shot batch inference on Tesla T4:
1. Loads yue-server and models directly from attached dataset: /kaggle/input/yue2-q8-models.
2. Launches local yue-server on 127.0.0.1:18087 (loopback only, zero public tunnels).
3. Submits task with prompt parameters.
4. Waits for synthesis to complete and saves output directly to /kaggle/working/output.mp3.
5. Sends completion announcement to ntfy channel.
6. Cleanly terminates and exits (0).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("yue2-batch")

# === TASK SPECIFICATION (injected per generation) ===
TASK_SPEC = {
    "title": "YuE2 Generated Track",
    "style": "Synthwave 80s, driving bass, analog synthesizer, punchy drums, guitar riff",
    "lyrics": "[Intro - Synth & Drums]\n[Verse 1]\nNeon lights in the night\n[Chorus]\nMusic plays through the dark",
    "duration": 120,
    "steps": 32,
    "seed": 42,
    "abc": None,
    "username": "jungterdai",
}
# ====================================================

INTERNAL_PORT = 18087


def find_dataset_files() -> tuple[Path, Path, Path, Path]:
    """Resolves yue-server binary, libraries, and models from dataset."""
    search_dirs = [
        Path("/kaggle/input/yue2-q8-models"),
        Path("/kaggle/input"),
    ]
    
    bin_path: Path | None = None
    backbone_path: Path | None = None
    vae_path: Path | None = None
    lib_dir: Path | None = None

    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.rglob("yue-server"):
            if p.is_file():
                bin_path = p
                lib_dir = p.parent
                break
        for p in d.rglob("YuE2-3B-Q8_0.gguf"):
            if p.is_file() and p.stat().st_size > 1024 * 1024:
                backbone_path = p
                break
        for p in d.rglob("YuE2-Vae-F32.gguf"):
            if p.is_file() and p.stat().st_size > 1024 * 1024:
                vae_path = p
                break

    if not bin_path or not backbone_path or not vae_path or not lib_dir:
        raise FileNotFoundError(
            f"Could not resolve all dataset files: bin={bin_path}, backbone={backbone_path}, vae={vae_path}"
        )

    # Ensure executable permission
    try:
        bin_path.chmod(0o755)
    except Exception:
        pass

    return bin_path, lib_dir, backbone_path, vae_path


def start_local_engine(
    bin_path: Path, lib_dir: Path, backbone_path: Path, vae_path: Path, port: int = INTERNAL_PORT
) -> subprocess.Popen[str]:
    """Launches yue-server strictly bound to localhost loopback."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    cuda_paths = [
        str(lib_dir),
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(cuda_paths) + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        str(bin_path),
        "--model", str(backbone_path),
        "--vae", str(vae_path),
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    logger.info("Launching local yue-server: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )


def wait_engine_ready(proc: subprocess.Popen[str], port: int = INTERNAL_PORT, timeout: int = 120) -> bool:
    """Waits for local engine to respond to /health."""
    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - start < timeout:
        if proc.poll() is not None:
            logger.error("Local yue-server crashed with exit code %s", proc.returncode)
            return False
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status in (200, 204):
                    logger.info("Local engine ready in %.1fs!", time.time() - start)
                    return True
        except Exception:
            time.sleep(1)
    return False


def run_synthesis(spec: dict, port: int = INTERNAL_PORT) -> bytes:
    """Submits synthesis task and waits for completed MP3 bytes."""
    synth_url = f"http://127.0.0.1:{port}/synth"
    payload = {
        "engine": "yue2",
        "style": spec.get("style", "Pop"),
        "lyrics": spec.get("lyrics", ""),
        "duration": float(spec.get("duration", 120)),
        "cot": "full",
        "steps": int(spec.get("steps", 32)),
        "output_format": "mp3",
        "mp3_bitrate": 320,
        "seed": int(spec.get("seed", 42)),
        "lm_seed": int(spec.get("seed", 42)),
    }
    if spec.get("abc"):
        payload["abc"] = spec["abc"]

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(synth_url, data=data, headers={"Content-Type": "application/json"})
    
    logger.info("Submitting task to engine: style=%s, duration=%ss", payload["style"], payload["duration"])
    with urllib.request.urlopen(req, timeout=30) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
        job_id = res_json.get("id") or res_json.get("job_id")
        if not job_id:
            raise RuntimeError(f"Engine did not return a valid job ID: {res_json}")

    logger.info("Job submitted with ID: %s. Monitoring synthesis progress...", job_id)
    
    # Poll job status
    poll_url = f"http://127.0.0.1:{port}/job?id={job_id}"
    start_poll = time.time()
    while time.time() - start_poll < 900:  # 15 minutes max
        time.sleep(2)
        try:
            p_req = urllib.request.Request(poll_url)
            with urllib.request.urlopen(p_req, timeout=10) as p_resp:
                st = json.loads(p_resp.read().decode("utf-8"))
                status = st.get("status", "running")
                progress = st.get("progress", 0)
                if status == "done":
                    logger.info("Synthesis complete! Downloading output audio...")
                    break
                elif status == "failed" or status == "error":
                    raise RuntimeError(f"Engine reported generation failure: {st}")
                logger.info("Generation in progress: status=%s, progress=%s%%", status, progress)
        except urllib.error.URLError as u_err:
            logger.warning("Poll warning: %s", u_err)

    # Download MP3
    result_url = f"http://127.0.0.1:{port}/job?id={job_id}&result=1"
    res_req = urllib.request.Request(result_url)
    with urllib.request.urlopen(res_req, timeout=60) as r_resp:
        audio_bytes = r_resp.read()
        logger.info("Retrieved %d audio bytes from engine.", len(audio_bytes))
        return audio_bytes


def notify_completion(username: str, title: str, audio_len: int) -> None:
    """Sends lightweight push notification on completion."""
    h = hashlib.sha256(username.strip().lower().encode("utf-8")).hexdigest()[:16]
    topic = f"yue2_studio_{h}"
    msg = json.dumps({"status": "complete", "title": title, "bytes": audio_len})
    for channel in ["ntfy.sh", "ntfy.envs.net"]:
        try:
            req = urllib.request.Request(
                f"https://{channel}/{topic}",
                data=msg.encode("utf-8"),
                headers={"Title": f"YuE2: {title} Ready!", "Priority": "high", "Tags": "musical_note"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass


def main() -> None:
    print("=" * 65)
    print("🎵 YuE2 Batch Synthesizer (Zero-Tunnel Architecture)")
    print("=" * 65)

    bin_path, lib_dir, backbone_path, vae_path = find_dataset_files()
    logger.info("Using engine: %s", bin_path)
    logger.info("Using models: %s | %s", backbone_path.name, vae_path.name)

    proc = start_local_engine(bin_path, lib_dir, backbone_path, vae_path, port=INTERNAL_PORT)
    try:
        ready = wait_engine_ready(proc, port=INTERNAL_PORT)
        if not ready:
            raise RuntimeError("Engine failed to become ready.")

        audio_bytes = run_synthesis(TASK_SPEC, port=INTERNAL_PORT)
        
        # Save to /kaggle/working/output.mp3
        out_file = Path("/kaggle/working/output.mp3")
        out_file.write_bytes(audio_bytes)
        logger.info("✓ Saved generated audio to: %s (%d bytes)", out_file, out_file.stat().st_size)

        notify_completion(
            username=TASK_SPEC.get("username", "jungterdai"),
            title=TASK_SPEC.get("title", "Generated Track"),
            audio_len=len(audio_bytes),
        )
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass
        logger.info("Engine cleanly stopped. Kaggle batch task complete.")


if __name__ == "__main__":
    main()
