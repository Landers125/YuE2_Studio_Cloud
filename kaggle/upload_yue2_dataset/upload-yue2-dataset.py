import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

print("=== Starting YuE2 Models & Engine Dataset Creator ===", flush=True)

# 1. Credentials setup for jungterdai
USERNAME = "jungterdai"
KEY = "a9c86217f94f6a3477c246d4c556a579"

os.environ["KAGGLE_USERNAME"] = USERNAME
os.environ["KAGGLE_KEY"] = KEY

kdir = Path.home() / ".kaggle"
kdir.mkdir(parents=True, exist_ok=True)
(kdir / "kaggle.json").write_text(
    json.dumps(
        {
            "username": USERNAME,
            "key": KEY,
        }
    )
)
os.chmod(kdir / "kaggle.json", 0o600)

# 2. System and Python tools
subprocess.check_call(["apt-get", "update", "-qq"])
subprocess.check_call(
    ["apt-get", "install", "-y", "-qq", "aria2", "cmake", "ninja-build", "git"]
)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "kaggle", "huggingface_hub"]
)

from kaggle.api.kaggle_api_extended import KaggleApi

out_dir = Path("/kaggle/working/yue2_dataset")
out_dir.mkdir(parents=True, exist_ok=True)

# 3. Build yue-server for Tesla T4 (sm_75)
work_dir = Path("/kaggle/working")
source_dir = work_dir / "yue2.cpp"
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
    print("Cloning yue2.cpp...", flush=True)
    subprocess.check_call(
        ["git", "clone", "--recurse-submodules", "-q", "https://github.com/timoncool/yue2.cpp.git", str(source_dir)]
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

print("Building yue-server binary...", flush=True)
subprocess.check_call(
    ["cmake", "--build", str(build_dir), "--config", "Release", "-j", str(os.cpu_count() or 4)]
)

built_binaries = list(build_dir.rglob("yue-server"))
if not built_binaries:
    raise FileNotFoundError("Compilation failed to create yue-server")

shutil.copy2(built_binaries[0], out_dir / "yue-server")
(out_dir / "yue-server").chmod(0o755)
print("Copied yue-server to dataset folder.", flush=True)

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

# 4. Download YuE2 GGUF models
models = [
    ("YuE2-3B-Q8_0.gguf", "https://huggingface.co/Serveurperso/YuE2-GGUF/resolve/64b030e3deb6e8150d2b7c0db641ef5a17eca8a3/YuE2-3B-Q8_0.gguf"),
    ("YuE2-Vae-F32.gguf", "https://huggingface.co/Serveurperso/YuE2-GGUF/resolve/64b030e3deb6e8150d2b7c0db641ef5a17eca8a3/YuE2-Vae-F32.gguf"),
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
DATASET_SLUG = "yue2-q8-models"
DATASET_REF = f"{USERNAME}/{DATASET_SLUG}"

metadata = {
    "title": "YuE2 Q8 Models and Engine",
    "id": DATASET_REF,
    "licenses": [{"name": "CC0-1.0"}],
}
(out_dir / "dataset-metadata.json").write_text(json.dumps(metadata))

api = KaggleApi()
api.authenticate()

user_datasets = [d.ref for d in api.dataset_list(user=USERNAME)]
print(f"User datasets for {USERNAME}: {user_datasets}", flush=True)

if DATASET_REF in user_datasets:
    print(f"Dataset already exists! Pushing new version...", flush=True)
    res = subprocess.run(
        [
            "kaggle",
            "datasets",
            "version",
            "-p", str(out_dir),
            "-m", "YuE2 Q8_0 models and yue-server binary",
            "--dir-mode", "skip",
        ],
        capture_output=True,
        text=True,
    )
    print("version result:", res.returncode, res.stdout, res.stderr, flush=True)
else:
    print(f"Creating new dataset {DATASET_REF}...", flush=True)
    res = subprocess.run(
        ["kaggle", "datasets", "create", "-p", str(out_dir), "--dir-mode", "skip"],
        capture_output=True,
        text=True,
    )
    print("create result:", res.returncode, res.stdout, res.stderr, flush=True)

print("=== YuE2 Dataset Creation Complete! ===", flush=True)
