import os
import sys
import shutil
import subprocess
import json
from pathlib import Path

print("=== Starting MuScriptor Large Dataset Creator ===", flush=True)

# 1. Credentials
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

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "kaggle", "huggingface_hub"]
)
from huggingface_hub import hf_hub_download
from kaggle.api.kaggle_api_extended import KaggleApi

out_dir = Path("/kaggle/working/muscriptor_dataset")
out_dir.mkdir(parents=True, exist_ok=True)

print("Downloading MuScriptor Large from HuggingFace (real files, no symlinks)...", flush=True)
hf_hub_download(
    repo_id="cocktailpeanut/muscriptor-large",
    filename="model.safetensors",
    local_dir=str(out_dir),
    local_dir_use_symlinks=False,
)
hf_hub_download(
    repo_id="cocktailpeanut/muscriptor-large",
    filename="config.json",
    local_dir=str(out_dir),
    local_dir_use_symlinks=False,
)

print("Checking downloaded files in destination folder:", flush=True)
for item in out_dir.iterdir():
    print(f"  {item.name}: is_file={item.is_file()}, is_symlink={item.is_symlink()}, size={item.stat().st_size} bytes", flush=True)

metadata = {
    "title": "MuScriptor Models",
    "id": "alexeyklochkov/muscriptor-models",
    "licenses": [{"name": "CC0-1.0"}],
}
(out_dir / "dataset-metadata.json").write_text(json.dumps(metadata))

api = KaggleApi()
api.authenticate()

print("Checking existing datasets...", flush=True)
user_datasets = [d.ref for d in api.dataset_list(user="alexeyklochkov")]
print(f"User datasets: {user_datasets}", flush=True)

if "alexeyklochkov/muscriptor-models" in user_datasets:
    print("Dataset already exists! Pushing new version...", flush=True)
    res = subprocess.run(
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(out_dir),
            "-m",
            "MuScriptor Large weights (5GB)",
            "--dir-mode",
            "skip",
        ],
        capture_output=True,
        text=True,
    )
    print("version result:", res.returncode, res.stdout, res.stderr, flush=True)
else:
    print("Creating new dataset alexeyklochkov/muscriptor-models...", flush=True)
    res = subprocess.run(
        ["kaggle", "datasets", "create", "-p", str(out_dir), "--dir-mode", "skip"],
        capture_output=True,
        text=True,
    )
    print("create result:", res.returncode, res.stdout, res.stderr, flush=True)

print("=== Dataset Creator Completed ===", flush=True)
