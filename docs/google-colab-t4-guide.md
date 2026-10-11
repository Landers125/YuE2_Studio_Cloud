# 🚀 YuE2 Studio на внешнем инстансе Google Colab (Tesla T4 16GB)

[English Guide below](#-english-guide) | [Русское руководство](#-русское-руководство)

---

## 🇷🇺 Русское руководство

Данное руководство объясняет, как запустить вычислительное ядро **YuE2 Studio** на бесплатном инстансе **Google Colab** с видеокартой **NVIDIA Tesla T4 (16 ГБ VRAM)** и подключить к нему ваш локальный интерфейс YuE2 Studio.

### 💡 Зачем это нужно?
- **Не требуется мощная локальная видеокарта**: Вы можете запускать генерацию музыки с ноутбука, офисного ПК или Mac даже без дискретного GPU.
- **Экономия дискового пространства**: Не нужно скачивать более 10 ГБ весов нейросети на локальный диск — все модели загружаются непосредственно в облако Google Colab.
- **Оптимизация под архитектуру Turing (`sm_75`)**: Бинарный файл `yue2.cpp` компилируется прямо в Colab с флагами оптимизации для чипа Tesla T4.
- **Полная функциональность**: Поддерживаются все режимы генерации (Text-to-song, Instrumental, Dual-track, SheetSage2 транскрипция для каверов), а также стриминг логов в реальном времени.

---

### 📋 Пошаговая инструкция

#### Шаг 1: Запуск Google Colab Notebook
1. Перейдите по ссылке на блокнот:
   👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timoncool/YuE2-Studio/blob/main/colab/YuE2_Studio_Colab_T4.ipynb)
2. В верхнем меню Google Colab выберите: **Среда выполнения (Runtime)** ➔ **Сменить среду выполнения (Change runtime type)**.
3. Убедитесь, что выбран **T4 GPU** (Аппаратный ускоритель: GPU, Тип GPU: T4). Нажмите **Сохранить**.
4. Запустите первую ячейку (`!nvidia-smi`), чтобы убедиться, что видеокарта Tesla T4 выделена.
5. Во второй ячейке выберите параметры:
   - **`QUANT`**: рекомендуем `Q8_0` (наилучшее качество 8-битного квантования, занимает ~3.55 ГБ) или `BF16` (оригинальная bfloat16 точность).
   - **`TUNNEL`**: `cloudflare` (рекомендуется — бесплатно, не требует регистрации и токенов).
   - **`KEEP_LOADED`**: `True` (оставляет веса в VRAM для мгновенных повторных генераций).
6. Запустите ячейку (кнопка `Play` или `Ctrl + Enter`).
7. Скрипт автоматически скачает необходимые системные библиотеки, скомпилирует `yue2.cpp` для `sm_75`, скачает квантованные веса моделей через `aria2c` и запустит туннель Cloudflare.
8. В выводе консоли появится баннер с публичным адресом:
   ```text
   ======================================================================
   🌟 PUBLIC TUNNEL URL:
   👉 https://abc-xyz-123.trycloudflare.com
   ======================================================================
   ```
   **Скопируйте этот URL** (включая `https://`).

---

#### Шаг 2: Подключение в YuE2 Studio
1. Запустите локальное приложение **YuE2 Studio** на вашем компьютере.
2. Если приложение открылось на экране первоначальной настройки (`SetupGate`):
   - Вы увидите баннер: *"Нет мощного GPU? Подключите Google Colab"*.
   - Нажмите кнопку **"Настроить Colab"** или откройте **Настройки (⚙️)** ➔ вкладка **Движок (Engine)**.
3. В настройках:
   - Включите тумблер: **Внешний движок / Google Colab (External / Google Colab Engine)**.
   - Вставьте скопированный адрес в поле **URL удаленного движка (Remote Engine URL)** (например, `https://abc-xyz-123.trycloudflare.com`).
   - Нажмите кнопку **Проверить связь (Test Connection)**.
4. После успешного подключения индикатор изменится на зелёный:
   > *«Удаленный движок готов к работе!»*
5. Нажмите **Сохранить** (или закройте модальное окно). Экран загрузки локальных моделей автоматически скроется, и вы сможете сразу сочинять треки!

---

### 🏗️ Архитектура решения

```
┌────────────────────────────────────────────────────────┐
│            Локальный ПК (Пользователь)                 │
│  - Настольное приложение YuE2 Studio (React + Rust)    │
│  - Локальный GPU не требуется                          │
│  - Локальные веса 10 ГБ не требуются                   │
└───────────────────────────┬────────────────────────────┘
                            │ HTTPS (REST API + SSE Logs)
                            ▼
┌────────────────────────────────────────────────────────┐
│              Cloudflare Quick Tunnel                   │
│        https://*.trycloudflare.com                     │
│  - Сквозное шифрование HTTPS                           │
│  - Поддержка Server-Sent Events (SSE) стриминга логов  │
│  - Поддержка скачивания больших аудиофайлов (WAV/MP3)  │
└───────────────────────────┬────────────────────────────┘
                            │ 127.0.0.1:18087
                            ▼
┌────────────────────────────────────────────────────────┐
│              Инстанс Google Colab                      │
│  - GPU: NVIDIA Tesla T4 (16 ГБ GDDR6 VRAM)             │
│  - yue2.cpp (скомпилирован с sm_75)                    │
│  - Веса: GGUF (Q8_0 / BF16) + VAE + SheetSage2         │
│  - yue-server с флагом --keep-loaded                   │
└────────────────────────────────────────────────────────┘
```

---

### 🔍 Распределение видеопамяти на Tesla T4 (15.3 ГБ)

| Компонент | Размер в VRAM | Пояснение |
|-----------|---------------|-----------|
| Контекст CUDA & yue2.cpp | ~500 МБ | Системный оверхед |
| AR Модель YuE2 (`Q8_0`) | ~3,550 МБ | Авторегрессионная модель (генерация токенов) |
| AR Модель YuE2 (`BF16`) | ~6,670 МБ | Полноточная модель (при выборе BF16) |
| NAR Модель / Companion | ~140 МБ | Неавторегрессионный декодер |
| VAE Vocoder Decoder | ~506 МБ | Преобразование латентных токенов в звук (WAV) |
| SheetSage2 Transcriber | ~913 МБ | Анализ вокала и аккордов для каверов |
| KV Cache (на 24,576 токенов) | ~4,500 МБ | Буфер внимания для длинных композиций |
| **ИТОГО** | **~9.6 ГБ (для Q8_0) / ~12.7 ГБ (для BF16)** | **Полностью укладывается в 15.3 ГБ VRAM!** |

---

### ❓ Частые вопросы и диагностика

1. **Что делать, если сессия Google Colab прервалась?**
   - Бесплатные сессии Colab могут перезапускаться при длительном простое. Просто нажмите `Среда выполнения` ➔ `Выполнить все`, дождитесь нового URL туннеля и обновите его в настройках YuE2 Studio.
2. **Как смотреть логи генерации с Colab?**
   - YuE2 Studio автоматически переключает стрим логов на удаленный сервер: все этапы генерации (Progress, Layer timings, KV-cache) видны во встроенном окне логов.
3. **Безопасен ли туннель Cloudflare?**
   - Да, Cloudflare Quick Tunnel создает временный уникальный криптографический поддомен со сквозным шифрованием TLS/HTTPS.

---

<br/>

## 🇬🇧 English Guide

This guide explains how to offload the **YuE2 Studio** music generation engine to a free **Google Colab** instance powered by an **NVIDIA Tesla T4 GPU (16 GB VRAM)** and seamlessly connect it with your local YuE2 Studio desktop application.

### 🌟 Key Advantages
- **No local GPU required**: Generate studio-grade music from any laptop or desktop.
- **Zero local storage overhead**: Avoid downloading 10+ GB of model weights locally; weights are pulled at multi-gigabit speeds directly into Google Colab.
- **Native Turing (`sm_75`) compilation**: `yue2.cpp` is compiled on-the-fly specifically for the Tesla T4 architecture.
- **1-Click Cloudflare Tunnel**: No account registration, API tokens, or port forwarding required.

---

### 🚀 Step-by-Step Instructions

#### Step 1: Launch Google Colab Notebook
1. Open the notebook:
   👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timoncool/YuE2-Studio/blob/main/colab/YuE2_Studio_Colab_T4.ipynb)
2. In Google Colab, verify GPU runtime: **Runtime** ➔ **Change runtime type** ➔ select **T4 GPU**.
3. Run **Step 1: Check GPU Environment** (`!nvidia-smi`) to verify 16 GB VRAM allocation.
4. In **Step 2: Configure & Launch Server**:
   - Set **`QUANT`**: `Q8_0` (recommended for optimal quality/speed) or `BF16`.
   - Set **`TUNNEL`**: `cloudflare` (zero config, free).
   - Set **`KEEP_LOADED`**: `True` (keeps models cached in VRAM for instant subsequent generations).
5. Run the cell and wait for the public URL banner:
   ```text
   ======================================================================
   🌟 PUBLIC TUNNEL URL:
   👉 https://abc-xyz-123.trycloudflare.com
   ======================================================================
   ```
   **Copy this URL**.

#### Step 2: Connect Local YuE2 Studio
1. Launch **YuE2 Studio** on your local machine.
2. Open **Settings (⚙️)** ➔ **Engine** tab.
3. Switch on **External / Google Colab Engine**.
4. Paste the Cloudflare URL into **Remote Engine URL**.
5. Click **Test Connection**. Once confirmed with a green badge, click **Save**.
6. You are ready to compose! All inference requests and audio decoding will run on the remote Tesla T4 GPU.
