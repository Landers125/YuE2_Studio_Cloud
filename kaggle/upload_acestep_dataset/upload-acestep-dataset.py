import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

print("=== Starting ACE-Step 1.5 Models & Engine Dataset Creator ===", flush=True)

# 1. Credentials setup
os.environ["KAGGLE_USERNAME"] = "alexeyklochkov"
os.environ["KAGGLE_KEY"] = "35a065781d57b9234a76deccd583da72"

kdir = Path.home() / ".kaggle"
kdir.mkdir(parents=True, exist_ok=True)
(kdir / "kaggle.json").write_text(
    json.dumps(
        {
            "username": "alexeyklochkov",
            "key": "35a065781d57b9234a76deccd583da72",
        }
    )
)
os.chmod(kdir / "kaggle.json", 0o600)

# 2. System and Python tools
subprocess.check_call(
    ["apt-get", "update", "-qq"]
)
subprocess.check_call(
    ["apt-get", "install", "-y", "-qq", "aria2", "cmake", "ninja-build", "git"]
)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "kaggle", "huggingface_hub"]
)

from kaggle.api.kaggle_api_extended import KaggleApi

out_dir = Path("/kaggle/working/acestep_dataset")
out_dir.mkdir(parents=True, exist_ok=True)

# 3. Build ace-server for Tesla T4 (sm_75)
work_dir = Path("/kaggle/working")
source_dir = work_dir / "acestep.cpp"
build_dir = source_dir / "build"

for src in [
    Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
    Path("/usr/local/cuda/lib64/stubs/libcuda.so"),
]:
    if src.exists():
        dst = Path("/usr/lib/x86_64-linux-gnu/libcuda.so")
        if not dst.exists():
            try:
                dst.symlink_to(src)
                print(f"CUDA stub symlink created: {dst} -> {src}", flush=True)
            except Exception:
                pass
        break

if not source_dir.exists():
    print("Cloning acestep.cpp (studio branch)...", flush=True)
    subprocess.check_call(
        ["git", "clone", "--recurse-submodules", "-b", "studio", "-q", "https://github.com/timoncool/acestep.cpp.git", str(source_dir)]
    )

print("Configuring CMake for Tesla T4 sm_75...", flush=True)
cmake_cmd = [
    "cmake",
    "-B", str(build_dir),
    "-S", str(source_dir),
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_CUDA=ON",
    "-DGGML_CUDA_NO_VMM=ON",
    "-DCMAKE_CUDA_ARCHITECTURES=75",
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    "-G", "Ninja",
]
for stub_dir in ["/usr/local/cuda/lib64/stubs", "/usr/lib/x86_64-linux-gnu"]:
    if Path(stub_dir).exists():
        cmake_cmd.append(f"-DCMAKE_LIBRARY_PATH={stub_dir}")
        break

subprocess.check_call(cmake_cmd)

print("Building ace-server binary...", flush=True)
subprocess.check_call(
    ["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(os.cpu_count() or 4), "--target", "ace-server"]
)

built_binaries = list(build_dir.rglob("ace-server"))
if not built_binaries:
    raise FileNotFoundError("Compilation failed to create ace-server")

shutil.copy2(built_binaries[0], out_dir / "ace-server")
(out_dir / "ace-server").chmod(0o755)
print("Copied ace-server to dataset folder.", flush=True)

# Copy all generated ggml shared libraries
for lib in build_dir.rglob("*.so*"):
    if lib.is_file() and not lib.is_symlink():
        try:
            shutil.copy2(lib, out_dir / lib.name)
        except Exception:
            pass
    elif lib.is_symlink():
        try:
            shutil.copy2(lib.resolve(), out_dir / lib.name)
        except Exception:
            pass
print("Copied shared libraries to dataset folder.", flush=True)

# 4. Download ACE-Step GGUF models
models = [
    ("acestep-v15-xl-turbo-Q8_0.gguf", "https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF/resolve/main/acestep-v15-xl-turbo-Q8_0.gguf"),
    ("acestep-5Hz-lm-1.7B-Q8_0.gguf", "https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF/resolve/main/acestep-5Hz-lm-1.7B-Q8_0.gguf"),
    ("Qwen3-Embedding-0.6B-Q8_0.gguf", "https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf"),
    ("vae-BF16.gguf", "https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF/resolve/main/vae-BF16.gguf"),
]

for filename, url in models:
    dest = out_dir / filename
    print(f"Downloading {filename}...", flush=True)
    subprocess.check_call(
        [
            "aria2c",
            "-c",
            "-x", "16",
            "-s", "16",
            "-k", "1M",
            "--file-allocation=none",
            "--dir", str(out_dir),
            "--out", filename,
            url,
        ]
    )
    print(f"Downloaded {filename}: {dest.stat().st_size} bytes", flush=True)

# 5. Dataset metadata and push
metadata = {
    "title": "ACE-Step Models and Engine",
    "id": "alexeyklochkov/acestep-models",
    "licenses": [{"name": "CC0-1.0"}],
}
(out_dir / "dataset-metadata.json").write_text(json.dumps(metadata))

api = KaggleApi()
api.authenticate()

user_datasets = [d.ref for d in api.dataset_list(user="alexeyklochkov")]
print(f"User datasets: {user_datasets}", flush=True)

if "alexeyklochkov/acestep-models" in user_datasets:
    print("Dataset already exists! Pushing new version...", flush=True)
    res = subprocess.run(
        [
            "kaggle",
            "datasets",
            "version",
            "-p", str(out_dir),
            "-m", "ACE-Step 1.5 XL Turbo Q8 models and engine binary",
            "--dir-mode", "skip",
        ],
        capture_output=True,
        text=True,
    )
    print("version result:", res.returncode, res.stdout, res.stderr, flush=True)
else:
    print("Creating new dataset alexeyklochkov/acestep-models...", flush=True)
    res = subprocess.run(
        ["kaggle", "datasets", "create", "-p", str(out_dir), "--dir-mode", "skip"],
        capture_output=True,
        text=True,
    )
    print("create result:", res.returncode, res.stdout, res.stderr, flush=True)

print("=== ACE-Step Dataset Creation Complete! ===", flush=True)
