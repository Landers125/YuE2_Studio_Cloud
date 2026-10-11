# YuE2 Studio — Google Colab Tesla T4 Backend

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timoncool/YuE2-Studio/blob/main/colab/YuE2_Studio_Colab_T4.ipynb)
[![GPU](https://img.shields.io/badge/GPU-Tesla%20T4%2016GB-76B900?logo=nvidia&logoColor=white)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Run the **YuE2 Studio** music generation engine on a free Google Colab **Tesla T4 GPU (16 GB VRAM)**. This allows you to run YuE2 Studio without requiring a powerful local NVIDIA GPU or downloading 10+ GB of model weights locally.

---

## ⚡ Quick Start

1. Open the notebook in Google Colab:
   👉 **[YuE2_Studio_Colab_T4.ipynb](https://colab.research.google.com/github/timoncool/YuE2-Studio/blob/main/colab/YuE2_Studio_Colab_T4.ipynb)**
2. In Google Colab, select **Runtime** -> **Change runtime type** -> **T4 GPU** (if not already selected).
3. Run the cells in order:
   - **Step 1**: Check GPU (`!nvidia-smi`)
   - **Step 2**: Configure options and start the engine
4. Copy the public Cloudflare tunnel URL printed at the end of the output:
   ```
   ======================================================================
   🌟 PUBLIC TUNNEL URL:
   👉 https://random-words.trycloudflare.com
   ======================================================================
   ```
5. In your local **YuE2 Studio** desktop app:
   - Click **Settings** (`⚙️`) in the navigation bar -> select the **Engine** tab.
   - Switch on **Google Colab / External Engine** (`Внешний движок / Google Colab`).
   - Paste the public URL into **Remote Engine URL**.
   - Click **Test Connection** (`Проверить связь`). Once verified, you're ready to create music!

---

## ⚙️ Architecture & Data Flow

```
┌──────────────────────────────────────┐
│       Local Machine (Client)         │
│  - YuE2 Studio (React + Rust Axum)   │
│  - No local GPU required             │
│  - No 10 GB weights download         │
└──────────────────┬───────────────────┘
                   │
                   │ HTTPS (REST API + SSE Logs)
                   ▼
┌──────────────────────────────────────┐
│       Cloudflare Quick Tunnel        │
│   https://*.trycloudflare.com        │
└──────────────────┬───────────────────┘
                   │
                   │ Localhost:18087
                   ▼
┌──────────────────────────────────────┐
│       Google Colab (Host)            │
│  - NVIDIA Tesla T4 (16 GB VRAM)      │
│  - yue2.cpp compiled for sm_75       │
│  - GGUF Quantized Models (Q8_0/BF16) │
│  - VAE + SheetSage2 Transcriber      │
└──────────────────────────────────────┘
```

---

## 📊 Tesla T4 VRAM Budget

The Tesla T4 GPU provides 15,360 MiB of VRAM:

| Component | Size in VRAM | Notes |
|-----------|--------------|-------|
| `yue2.cpp` Base CUDA context | ~500 MB | Runtime overhead |
| YuE2 AR Model (`Q8_0`) | ~3,550 MB | Recommended balance of fidelity and speed |
| YuE2 AR Model (`BF16`) | ~6,670 MB | Full precision (fits within 16 GB with headroom) |
| YuE2 NAR Model / Companion | ~140 MB | Fast parallel decoding |
| VAE Vocoder Decoder | ~506 MB | Generates final 32-bit float audio |
| SheetSage2 Transcriber | ~913 MB | Cover audio transcription |
| KV Cache (24,576 tokens) | ~4,500 MB | Context memory for full songs |
| **Total VRAM Consumption** | **~9.6 GB (Q8_0) / ~12.7 GB (BF16)** | Well within the 15.3 GB limit ✅ |

---

## 🎛️ CLI Options (`run_colab_server.py`)

You can also run the script directly in any Linux / Colab terminal:

```bash
python run_colab_server.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--quant` | `Q8_0` | GGUF Quantization: `Q8_0`, `BF16`, `Q6_K`, `Q5_K_M` |
| `--tunnel` | `cloudflare` | Tunnel provider: `cloudflare` (no auth needed) or `ngrok` |
| `--ngrok-token` | `None` | Ngrok authtoken (if using ngrok) |
| `--keep-loaded` | `True` | Keep weights in GPU memory across requests |
| `--port` | `18087` | Port for `yue-server` |
| `--model-dir` | `/content/models` | Storage path for downloaded GGUF weights |
| `--workspace` | `/content/yue2_build` | Build directory for `yue2.cpp` |
