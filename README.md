# 🚀 LLM Local Stack - Local AI Infrastructure

This documentation describes the implementation of a complete AI stack running locally via **WSL2 (Ubuntu)**. The ecosystem uses `llama.cpp` for high-performance CPU inference, a **FastAPI Gateway** for model routing, and management scripts to facilitate day-to-day usage.

---

## 📁 0. Real Folder Structure

### Installation

**Clone the repository:**
  ```bash
  git clone https://github.com/denilsonbomjesus/local-llm-cpu-stack.git
  cd local-llm-cpu-stack
  ```

> Note: It is recommended to place the project in the root of the Linux user profile. After downloading/cloning the repository, rename the main directory from "local-llm-cpu-stack" to "llm-stack" to better fit the commands.

The project is organized to keep models, servers, and scripts isolated:

```bash
~/llm-stack
  ├── llama.cpp/               # Source code and compiled binaries
  ├── models/                  # .gguf files organized by category
  │   ├── text/                # Qwen2.5-1.5B, Gemma 2 2B, Gemma 4 (E2B/E4B), Dolphin3.0
  │   ├── code/                # Qwen2.5-Coder, Ministral-3-3B-Instruct-2512
  │   ├── vision/              # Qwen2.5-VL (Model + mmproj)
  │   └── bonsai/              # Bonsai 27B 1-bit + Ternary Bonsai 27B
  ├── vision/                  # Python server for Qwen-VL
  ├── gateway/                 # FastAPI router (Port 9000)
  ├── scripts/                 # Control scripts (manage.sh, chat.sh)
  ├── logs/                    # Execution logs (optional)
  ├── external_config.yaml     # Model menu for external tools
  └── README.md                # This documentation
```

---

## 📦 1. Base installation on WSL2 (Ubuntu)
### 1.1. System dependencies
```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
  build-essential cmake git pkg-config \
  libssl-dev zlib1g-dev \
  python3 python3-venv python3-pip \
  curl wget \
  tmux \
  jq

# OpenCL (obrigatório para compilar com OpenVINO e usar a GPU Intel Iris Xe)
sudo apt install -y \
  ocl-icd-opencl-dev opencl-headers opencl-clhpp-headers intel-opencl-icd
```

This covers compilation of `llama.cpp` (com ou sem OpenVINO), Python for gateway/vision, `tmux` to run in the background. [github](http://github.com/ggml-org/llama.cpp)

If you want Node.js (optional, not used here):

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

***
## 🧠 2. Clone and compile `llama.cpp` (CPU-only + OpenVINO opcional)
We'll use the official repository `ggml-org/llama.cpp`. [github](http://github.com/ggml-org/llama.cpp)
### 2.1. Clone
```bash
cd ~/llm-stack
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```
### 2.2. Build otimizado (Release, sem aceleração adicional)
```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j"$(nproc)"
```

This generates binaries in `~/llm-stack/llama.cpp/build/` (`llama-cli`, `llama-server`, etc.). [github](http://github.com/ggml-org/llama.cpp)

> Note: we are not enabling CUDA/CL here to keep things simple and CPU-focused; performance on the chosen models is excellent even without dedicated GPU acceleration.

### 2.3. Build com OpenVINO (para Intel CPU/GPU — RECOMENDADO para seu hardware)

Se você tem um processador Intel (especialmente 11ª/12ª/13ª geração ou Intel Core Ultra) com placa integrada Intel Iris Xe, esta opção oferece:
- **Prompt processing ~2-3x mais rápido** (primeira resposta chega mais rápido)
- **Token generation 10-30% mais rápido** (texto gerado por segundo)
- **Suporte a AVX-VNNI** (instruções vetoriais otimizadas da Intel)
- **Offloading para Iris Xe iGPU** via OpenCL

#### 2.3.1. Instalar OpenVINO Runtime
```bash
cd ~
# Download do OpenVINO 2026.2.1 para Ubuntu 24.04
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.2.1/linux/openvino_toolkit_ubuntu24_2026.2.1.21919.ede283a88e3_x86_64.tgz -O openvino.tgz

# Extrair para /opt/intel/
sudo mkdir -p /opt/intel/openvino_2026.2.1
sudo tar -xzf openvino.tgz -C /opt/intel/openvino_2026.2.1 --strip-components=1
sudo ln -sfn /opt/intel/openvino_2026.2.1 /opt/intel/openvino
rm openvino.tgz

# Instalar dependências de runtime do OpenVINO
sudo -E /opt/intel/openvino/install_dependencies/install_openvino_dependencies.sh -y
```

#### 2.3.2. Compilar llama.cpp com OpenVINO
```bash
cd ~/llm-stack/llama.cpp

# Ativar ambiente OpenVINO
source /opt/intel/openvino/setupvars.sh

# Limpar build anterior (se existir)
rm -rf build

# Configurar com CMake (usando Ninja para compilação mais rápida)
cmake -B build/ReleaseOV -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENVINO=ON

# Compilar (12 threads)
cmake --build build/ReleaseOV --parallel
```

Isso gera binários em `~/llm-stack/llama.cpp/build/ReleaseOV/bin/`.

#### 2.3.3. Adicionar ao ~/.bashrc (automático)
Adicione estas linhas ao final do `~/.bashrc`:

```bash
# === OpenVINO ===
if [ -f /opt/intel/openvino/setupvars.sh ]; then
    source /opt/intel/openvino/setupvars.sh
fi
# Usar GPU (Iris Xe) por padrão. Fallback para CPU se GPU indisponível
export GGML_OPENVINO_DEVICE="${GGML_OPENVINO_DEVICE:-GPU}"
```

> ⚠️ **Importante sobre modelos Qwen**: Os modelos Qwen (Qwen2.5, Qwen-Coder, Qwen-VL) têm uma limitação conhecida no backend OpenVINO: a execução **stateful na GPU falha**. Os scripts `manage.sh` e `chat.sh` já incluem uma detecção automática que define `GGML_OPENVINO_STATEFUL_EXECUTION=0` para modelos Qwen e `=1` para os demais (Ministral, Gemma).

***
## 📥 3. Download models (WSL2)
We will standardize everything in `~/llm-stack/models`.

```bash
mkdir -p ~/llm-stack/models/{text,code,vision}
cd ~/llm-stack/models
```
### 3.1. Text / agents

#### Dolphin3.0-Llama3.2-3B (Q4_K_M)

O **Dolphin3.0** é um modelo da Cognitive Computations baseado no Llama 3.2-3B, conhecido por ser uncensored e versátil. Ótimo para tarefas gerais, chatbot e código simples.

Repo: `dphn/Dolphin3.0-Llama3.2-3B` (safetensors original). GGUF convertido por `bartowski/Dolphin3.0-Llama3.2-3B-GGUF`. [huggingface](https://huggingface.co/bartowski/Dolphin3.0-Llama3.2-3B-GGUF)

- **Arquivo:** `Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf` (1.9 GB)
- **Qualidade:** Excelente para chat geral, código simples, tarefas do dia a dia
- **RAM (4K ctx):** ~2.9 GB ✅ cabe em 8 GB WSL2
- **Velocidade CPU (i5):** ~20-30 tok/s ⚡ Muito rápido
- **Sem thinking mode** (Llama 3.2 base, sem tokens especiais de raciocínio)

```bash
cd ~/llm-stack/models/text

wget -O Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Dolphin3.0-Llama3.2-3B-GGUF/resolve/main/Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf
```

#### Qwen2.5‑1.5B‑Instruct‑Q4_K_M.gguf

Official GGUF repo: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`. [huggingface](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)

File:

- `qwen2.5-1.5b-instruct-q4_k_m.gguf` (4‑bit K‑quant, recommended for laptops). [skywork](https://skywork.ai/blog/models/qwen2-5-1-5b-instruct-gguf-free-chat-online-skywork-ai/)

```bash
cd ~/llm-stack/models/text
wget -O qwen2.5-1.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

#### Gemma-2-2b-it-abliterated-q4_k_m.gguf

Repo: `bartowski/gemma-2-2b-it-abliterated-GGUF`. [huggingface](https://huggingface.co/bartowski/gemma-2-2b-it-abliterated-GGUF/resolve/main/gemma-2-2b-it-abliterated-Q4_K_M.gguf)

File:

- `Gemma-2-2b-it-abliterated-gguf` (~1.6 GB). [huggingface](https://huggingface.co/bartowski/gemma-2-2b-it-abliterated-GGUF/resolve/main/gemma-2-2b-it-abliterated-Q4_K_M.gguf)

```bash
cd ~/llm-stack/models/text
wget -O gemma-2-2b-it-abliterated-q4_k_m.gguf \
  https://huggingface.co/bartowski/gemma-2-2b-it-abliterated-GGUF/resolve/main/gemma-2-2b-it-abliterated-Q4_K_M.gguf
```
### 3.2. Code
#### Qwen2.5‑Coder‑1.5B‑Instruct‑Q4_K_M.gguf

GGUF repo: `bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF`. [huggingface](https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF)

Recommended file: `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf` (4‑bit, “default size for most use cases, *recommended*”). [huggingface](https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF)

```bash
cd ~/llm-stack/models/code
wget -O qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf
```

#### Ministral-3-3b-instruct-2512-q4_k_m.gguf

Repo: `unsloth/Ministral-3-3B-Instruct-2512-GGUF`. [huggingface](https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf)

File:

- `Ministral-3-3B-Instruct-2512-GGUF` (Q4_K_M = 4.5 bpw, “balanced quality – recommended”). [huggingface](https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf)

```bash
cd ~/llm-stack/models/code
wget -O ministral-3-3b-instruct-2512-q4_k_m.gguf \
  https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
```

#### Download mmproj for Ministral
This file is the "eyes" of Ministral-3-3B-Instruct-2512.

Run this command:
```bash
wget -O ~/llm-stack/models/code/ministral-3-3b-instruct-2512-mmproj-f16.gguf \
  https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/resolve/main/mmproj-F16.gguf
```

### 3.3. Vision

#### Qwen2.5‑VL‑3B‑Abliterated‑Caption‑it (GGUF)

Repo: `prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF`. [huggingface](https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF)

3B file available in the GGUF tree:

- `Qwen2.5-VL-3B-Abliterated-Caption-it.IQ4_XS.gguf` (4‑bit IQ4_XS, optimized, still 4‑bit; and explicitly Abliterated / Uncensored). [huggingface](https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF/tree/main/Qwen2.5-VL-3B-Abliterated-Caption-it-GGUF)

```bash
cd ~/llm-stack/models/vision
wget -O qwen2.5-vl-3b-abliterated-caption-it-iq4_xs.gguf \
  https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF/resolve/main/Qwen2.5-VL-3B-Abliterated-Caption-it-GGUF/Qwen2.5-VL-3B-Abliterated-Caption-it.IQ4_XS.gguf
```

#### Download mmproj for Qwen2.5-VL-3B
This file is the "eyes" of Qwen2.5-VL-3B.

Run this command:
```bash
wget -O ~/llm-stack/models/vision/qwen2.5-vl-3b-abliterated-caption-it.mmproj-Q8_0.gguf \
  https://huggingface.co/lmstudio-community/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-model-f16.gguf
```

> Note: I didn't see Q4_K_M explicitly in the tree, but IQ4_XS is a compact 4‑bit variant, suitable for CPU. [huggingface](https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF)

### 3.4. Bonsai 27B (1-bit & Ternary)

Dois modelos revolucionários da PrismML que cabem no seu setup. O Bonsai 27B é baseado no Qwen3.6-27B com pesos em formato binário (1-bit) ou ternário, atingindo ~90-95% da qualidade FP16 com uma fração do tamanho.

**Pré-requisito:** llama.cpp atualizado (versão >= que suporta Q1_0 e Q2_0_g64).

#### Bonsai 27B 1-bit (Q1_0_g128)

Repo: `prism-ml/Bonsai-27B-gguf`. [huggingface](https://huggingface.co/prism-ml/Bonsai-27B-gguf)

- **Arquivo:** `Bonsai-27B-Q1_0.gguf` (3.8 GB)
- **Qualidade:** 89.5% do FP16 (média 76.11 nos benchmarks)
- **RAM (4K ctx):** ~5.2 GB ✅ cabe em 8 GB WSL2
- **RAM (10K ctx):** ~5.6 GB ✅ 
- **RAM (100K ctx):** ~6.8 GB (com KV cache 4-bit)
- **Velocidade estimada (CPU):** ~4-8 tok/s

```bash
mkdir -p ~/llm-stack/models/bonsai
cd ~/llm-stack/models/bonsai

wget -O Bonsai-27B-Q1_0.gguf \
  https://huggingface.co/prism-ml/Bonsai-27B-gguf/resolve/main/Bonsai-27B-Q1_0.gguf
```

#### Ternary Bonsai 27B (Q2_0_g64)

Repo: `prism-ml/Ternary-Bonsai-27B-gguf`. [huggingface](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)

- **Arquivo:** `Ternary-Bonsai-27B-Q2_g64.gguf` (7.2 GB) — variante mainline-compatible (grupo 64)
- **Qualidade:** 94.6% do FP16 (média 80.49) — **quase perfeito**
- **RAM (4K ctx):** ~8.4 GB ⚠️ precisa de 10 GB WSL2
- **RAM (10K ctx):** ~8.7 GB ⚠️
- **Velocidade estimada (CPU):** ~2-5 tok/s

```bash
cd ~/llm-stack/models/bonsai

wget -O Ternary-Bonsai-27B-Q2_g64.gguf \
  https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/resolve/main/Ternary-Bonsai-27B-Q2_g64.gguf
```

> **Sobre MTP/DSpark:** Ambos os modelos vêm com DSpark drafter, mas ele **não acelera em CPU** (só CUDA). Pule o download do drafter para economizar ~1.8 GB de RAM.

> **Sobre o llama.cpp:** O Q1_0 e Q2_0_g64 já estão **mergeados no mainline** do llama.cpp. Basta dar `git pull` e recompilar o build CPU para suportá-los.

### 3.5. Gemma 4 (E2B & E4B)

Os modelos **Gemma 4** da Google representam um novo patamar de eficiência com a arquitetura PLE (Pyramid-Like Expansion). São modelos multimodal (texto + visão) com suporte nativo a GGUF.

> **Pré-requisito:** llama.cpp versão b10173+ (mainline recente). Ambos usam `qat-q4_0`.

#### Gemma 4 E2B (2.3B efetivos)

Repo: `google/gemma-4-E2B-it-qat-q4_0-gguf`. [huggingface](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)

- **Arquivo:** `gemma-4-E2B_q4_0-it.gguf` (3.2 GB)
- **Qualidade:** Excelente para conversação, resumos e classificação
- **RAM (4K ctx, texto puro):** ~4.2 GB ✅ cabe em 8 GB WSL2
- **RAM (4K ctx, com mmproj):** ~5.2 GB ✅
- **Velocidade CPU (i5):** ~22-35 tok/s ⚡ Muito rápido
- **mmproj:** `gemma-4-E2B-it-mmproj.gguf` (942 MB, para visão)

**Download (arquivos já em models/text/):**
```bash
cd ~/llm-stack/models/text

# Modelo principal (obrigatório)
wget -O gemma-4-E2B_q4_0-it.gguf \
  https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf

# mmproj (opcional — só se for usar visão)
wget -O gemma-4-E2B-it-mmproj.gguf \
  https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B-it-mmproj.gguf
```

#### Gemma 4 E4B (4.5B efetivos)

Repo: `google/gemma-4-E4B-it-qat-q4_0-gguf`. [huggingface](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf)

- **Arquivo:** `gemma-4-E4B_q4_0-it.gguf` (4.9 GB)
- **Qualidade:** Superior em lógica, programação e raciocínio
- **RAM (4K ctx, texto puro):** ~5.9 GB ⚠️ apertado em 8 GB WSL2
- **RAM (4K ctx, com mmproj):** ~6.9 GB ⚠️ recomendado 10 GB WSL2
- **Velocidade CPU (i5):** ~10-15 tok/s
- **mmproj:** `gemma-4-E4B-it-mmproj.gguf` (946 MB, para visão)

**Download:**
```bash
cd ~/llm-stack/models/text

# Modelo principal
wget -O gemma-4-E4B_q4_0-it.gguf \
  https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B_q4_0-it.gguf

# mmproj (opcional)
wget -O gemma-4-E4B-it-mmproj.gguf \
  https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf/resolve/main/gemma-4-E4B-it-mmproj.gguf
```

> **Dica de RAM:** Use as variantes **sem mmproj** (gemma4-e2b, gemma4-e4b) para economizar ~1 GB quando não for usar imagens. As variantes **-vision** (gemma4-e2b-vision, gemma4-e4b-vision) carregam o mmproj para suporte multimodal.

> **Sobre thinking mode:** Gemma 4 tem pensamento interno via `<|think|>` tokens. Use as variantes `-nothink` para desligar e obter respostas mais diretas.

***
## 🧠 2b. What is Q4_K_M / IQ4_XS?
- **Q4_K_M**: 4‑bit quantization in "super‑blocks" with per‑block statistics, 4.5 bits per weight, optimal quality vs RAM trade‑off – the normally recommended option for general use. [huggingface](https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF)
- **IQ4_XS**: "imatrix" super‑compact 4‑bit variant, with additional compression and small quality loss, often used in small models to reduce footprint. [huggingface](https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF)

Alternatives:

- **Q5_K_M** (when available) → +quality, +RAM, +latency. [huggingface](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
- **Q8_0** → almost no loss, but practically doubles memory and cost. [inference.readthedocs](https://inference.readthedocs.io/en/v0.15.4/models/builtin/llm/qwen2.5-instruct.html)

For your notebook, Q4_K_M / IQ4_XS is the sweet spot. [skywork](https://skywork.ai/blog/models/qwen2-5-1-5b-instruct-gguf-free-chat-online-skywork-ai/)

***
## 🚀 4. Running models with `manage.sh`
To simplify the management of multiple models, we use the script `./scripts/manage.sh`. It automates the creation of `tmux` sessions and ensures each model runs on the correct port.

### 4.1. How to start a model
In the terminal (WSL2), run:
```bash
./scripts/manage.sh start [model]
```

**Available options:**
- `qwen-text`          → Port 8001 (Qwen 2.5 1.5B)
- `gemma2`             → Port 8002 (Gemma 2 2B)
- `qwen-coder`         → Port 8003 (Qwen Coder 1.5B)
- `ministral-agent`    → Port 8004 (Ministral 3 3B)
- `ministral-vision`   → Port 8005 (Ministral 3 3B)
- `vision`             → Port 8010 (Qwen-VL Python Server)
- `bonsai-27b`         → Port 8011 (Bonsai 27B 1-bit ~4-8 tok/s)
- `ternary-bonsai`     → Port 8012 (Ternary Bonsai 27B ~2-5 tok/s)
- `gemma4-e2b`        → Port 8021 (Gemma 4 E2B 2.3B ~22-35 tok/s, texto puro)
- `gemma4-e2b-nothink`→ Port 8023 (Gemma 4 E2B, thinking OFF)
- `gemma4-e2b-vision` → Port 8031 (Gemma 4 E2B + visão)
- `gemma4-e2b-vision-nothink`→ Port 8033 (Gemma 4 E2B + visão, thinking OFF)
- `gemma4-e4b`        → Port 8022 (Gemma 4 E4B 4.5B ~10-15 tok/s, texto puro)
- `gemma4-e4b-nothink`→ Port 8024 (Gemma 4 E4B, thinking OFF)
- `dolphin3`          → Port 8041 (Dolphin3.0 Llama3.2-3B Q4_K_M ~20-30 tok/s)
- `gemma4-e4b-vision` → Port 8032 (Gemma 4 E4B + visão)
- `gemma4-e4b-vision-nothink`→ Port 8034 (Gemma 4 E4B + visão, thinking OFF)
- `gateway`            → Port 9000 (The Central Router)

### 4.2. Useful Manager commands
- `./scripts/manage.sh status`      → See what is running.
- `./scripts/manage.sh stop-all`    → Kill all services and free RAM.
- `./scripts/manage.sh stop gemma2` → Stop only a specific model.

***
## 🌐 5. OpenAI‑like HTTP API of `llama-server`
The `llama-server` exposes OpenAI‑compatible endpoints, such as `/v1/chat/completions`. [github](http://github.com/ggml-org/llama.cpp)

Quick test (on WSL2):

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct-q4_k_m",
    "messages": [
      {"role": "user", "content": "Summarize in 3 points the function of an operating system."}
    ],
    "max_tokens": 128
  }'
```

This should already return JSON in OpenAI format. [learn.arm](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama-cpu/llama-server/)

> The `"model"` field here is ignored by `llama-server` (it is already "fixed" in the binary), but we will use this field in the *gateway* for routing.

***
## 🧠 10. Vision – servers
### 10.1. Qwen2.5‑VL‑3B‑Abliterated (uncensored caption, on WSL2)
For vision, there is currently no direct multimodal support for Qwen‑VL in vanilla `llama.cpp`, so we will create a micro‑service in Python using Hugging Face Transformers, running CPU‑only (fast enough for occasional images). [huggingface](https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF)

#### 10.1.1. Python vision environment (WSL2)

```bash
cd ~/llm-stack/vision
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install "transformers>=4.40.0" "accelerate" "torch" "safetensors" pillow fastapi uvicorn[standard]
```

> Plain Torch CPU is sufficient for your case; if you want to optimize later, you can swap the backend.

#### 10.1.2. FastAPI server for Qwen‑VL Abliterated

`~/llm-stack/vision/qwen_vl_server.py`:

```python
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import io

from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = "prithivMLmods/Qwen2.5-VL-3B-Abliterated-Caption-GGUF"

app = FastAPI()

print("Loading Qwen2.5-VL-3B-Abliterated model...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, trust_remote_code=True)

class CaptionRequest(BaseModel):
    prompt: Optional[str] = "Describe this image in detail."

@app.post("/v1/images/captions")
async def caption_image(request: CaptionRequest = None, file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prompt = (request.prompt if request and request.prompt else
              "Describe this image in detail.")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=128)
    caption = processor.batch_decode(out, skip_special_tokens=True)[0]
    return {"caption": caption}
```

> Note: the base model from the repo is Qwen2.5‑VL Abliterated; the GGUF file is used for `llama.cpp`, but here we use the original HF model (non‑GGUF) from the same repo; this gives you a dedicated caption endpoint. [huggingface](https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF/tree/main/Qwen2.5-VL-3B-Abliterated-Caption-it-GGUF)

Run:

```bash
cd ~/llm-stack/vision
source venv/bin/activate
tmux new-session -d -s qwen-vl \
  "uvicorn qwen_vl_server:app --host 0.0.0.0 --port 8010"
```

Test:

```bash
curl -X POST "http://localhost:8010/v1/images/captions" \
  -F "file=@/path/to/your/image.jpg"
```

***

## 🌐 6. Gateway / Model Router (WSL2, FastAPI)
We will create an OpenAI‑style gateway in **WSL2** that exposes `/v1/chat/completions` and forwards to the correct server based on the `"model"` field. [llamastack.github](https://llamastack.github.io/docs/providers/openai)
### 6.1. Python environment for gateway
```bash
cd ~/llm-stack/gateway
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install fastapi uvicorn[standard] httpx pyyaml
```
### 6.2. YAML model config (for n8n and router)
`~/llm-stack/gateway/models.yaml`:

```yaml
models:
  qwen2.5-1.5b-instruct:
    type: text
    endpoint: http://localhost:8001

  gemma-2-2b-abliterated:
    type: text
    endpoint: http://localhost:8002

  qwen2.5-coder-1.5b:
    type: code
    endpoint: http://localhost:8003

  ministral-agent:
    type: code
    endpoint: http://localhost:8004

  ministral-vision:
    type: vision
    endpoint: http://localhost:8005

  qwen2.5-vl-3b-abliterated:
    type: vision
    endpoint: http://localhost:8010

  bonsai-27b-1bit:
    type: text
    endpoint: http://localhost:8011

  ternary-bonsai-27b:
    type: text
    endpoint: http://localhost:8012

  gemma-4-e2b:
    type: text
    endpoint: http://localhost:8021

  gemma-4-e2b-nothink:
    type: text
    endpoint: http://localhost:8023

  gemma-4-e2b-vision:
    type: vision
    endpoint: http://localhost:8031

  gemma-4-e2b-vision-nothink:
    type: vision
    endpoint: http://localhost:8033

  gemma-4-e4b:
    type: text
    endpoint: http://localhost:8022

  gemma-4-e4b-nothink:
    type: text
    endpoint: http://localhost:8024

  gemma-4-e4b-vision:
    type: vision
    endpoint: http://localhost:8032

  gemma-4-e4b-vision-nothink:
    type: vision
    endpoint: http://localhost:8034
```

> This same YAML can be reused in n8n to have a centralized endpoint table.
### 6.3. FastAPI Gateway – OpenAI‑style router
`~/llm-stack/gateway/gateway_server.py`:

```python
import json
import yaml
import httpx
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse # Added StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

CONFIG_PATH = "models.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

MODEL_MAP: Dict[str, Dict[str, Any]] = CONFIG["models"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/models")
async def list_models():
    models_list = []
    for model_id in MODEL_MAP.keys():
        models_list.append({
            "id": model_id,
            "object": "model",
            "created": 1677610602,
            "owned_by": "local-gateway"
        })
    return {"object": "list", "data": models_list}

def get_backend(model_name: str) -> Dict[str, Any]:
    if model_name not in MODEL_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'")
    return MODEL_MAP[model_name]

# HELPER FUNCTION FOR STREAMING
# In your proxy_stream, try to ensure that iter_bytes is read without aggressive buffering
async def proxy_stream(url: str, body: dict):
    client = httpx.AsyncClient(timeout=None)
    async def event_generator():
        try:
            async with client.stream("POST", url, json=body) as response:
                # The secret is to read chunks as they arrive
                async for chunk in response.aiter_raw(): 
                    yield chunk
        finally:
            await client.aclose()
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_name = body.get("model")

    if not model_name:
        raise HTTPException(status_code=400, detail="'model' field is required")

    backend = get_backend(model_name)
    endpoint = backend["endpoint"]

    # --- SANITIZE BODY FOR AGENTS (Avoids "Empty Response") ---
    # Keeps OpenWebUI and n8n working, but removes keys that llama-server rejects
    allowed_keys = {
        "model", "messages", "temperature", "top_p", "stream", 
        "max_tokens", "stop", "presence_penalty", "frequency_penalty"
    }
    clean_body = {k: v for k, v in body.items() if k in allowed_keys}
    # --------------------------------------------------------------------

    if backend["type"] in ("text", "code", "vision"):
        # If the client requests streaming, we forward the stream using the clean body
        if body.get("stream", False):
            return await proxy_stream(f"{endpoint}/v1/chat/completions", clean_body)

        # Otherwise, normal response using the clean body
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(f"{endpoint}/v1/chat/completions", json=clean_body)
            return JSONResponse(status_code=resp.status_code, content=resp.json())

    raise HTTPException(status_code=500, detail="Unsupported backend type")

@app.post("/v1/images/captions/{model_name}")
async def vision_captions(model_name: str, request: Request):
    backend = get_backend(model_name)
    endpoint = backend["endpoint"]
    async with httpx.AsyncClient(timeout=None) as client:
        content_type = request.headers.get("Content-Type")
        body = await request.body()
        resp = await client.post(
            f"{endpoint}/v1/images/captions",
            content=body,
            headers={"Content-Type": content_type}
        )
    return JSONResponse(status_code=resp.status_code, content=resp.json())
```

> I kept `/v1/chat/completions` centralized for text/code (where OpenAI compatibility matters most for n8n/AnythingLLM/etc.). For vision, I recommend calling the vision endpoint directly (since each has a different format). [github](http://github.com/ggml-org/llama.cpp)

Run the gateway:

```bash
cd ~/llm-stack/gateway
source venv/bin/activate
tmux new-session -d -s router \
  "uvicorn gateway_server:app --host 0.0.0.0 --port 9000"
```

***
## 🔗 8. Example consumption via HTTP (curl)
### 8.1. Chat with Qwen‑Instruct (via gateway)
```bash
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-1.5b-instruct",
    "messages": [
      {"role": "system", "content": "You are a concise technical assistant."},
      {"role": "user", "content": "Briefly explain what a syscall is."}
    ],
    "max_tokens": 128,
    "temperature": 0.4
  }' | jq
```
### 8.2. Chat with Gemma-2-2b-abliterated
```bash
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-2-2b-abliterated",
    "messages": [
      {"role": "user", "content": "Let's talk about distributed systems design."}
    ],
    "max_tokens": 256
  }' | jq
```
### 8.3. Code assistant (Qwen‑Coder) on the same endpoint
```bash
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-1.5b",
    "messages": [
      {"role": "system", "content": "You are a coding assistant. Respond with code and brief explanation."},
      {"role": "user", "content": "Write a C function that reverses a string in-place."}
    ],
    "max_tokens": 256
  }' | jq
```

***
## ⚙️ 9. Practical optimization (`-t`, `-b`, `-ngl`, memory)
**`-t` (threads)**  
- Start with `8`. If you see the CPU does not reach 100% on all cores, you can increase to `10`.  
- Too high → scheduling overhead without real gain.

**`-b` (batch size)**  
- 256 is safe; trying `-b 512` can increase tok/s, but consumes more RAM.  
- Monitor with `htop` / `time`.

**`-ngl` (GPU layers)**  
- 0 (CPU only) is the default here; the total focus is on performance via CPU vector instructions in WSL2.

**Memory per model** (approx):

- Qwen2.5‑1.5B‑Instruct‑Q4_K_M → ~1.1 GB + KV‑cache (up to ~2–3 GB with large context). [huggingface](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
- Qwen2.5‑Coder‑1.5B‑Q4_K_M → ~1.0 GB + cache. [huggingface](https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF)
- **Bonsai 27B 1-bit** → ~3.9 GB + ~1.3 GB overhead + KV cache (~5.2 GB total at 4K ctx)
- **Ternary Bonsai 27B** → ~7.2 GB + ~1.2 GB overhead + KV cache (~8.4 GB total at 4K ctx)
- **Gemma 4 E2B (texto)** → ~3.2 GB + ~1 GB overhead + KV cache (~4.2 GB total at 4K ctx)
- **Gemma 4 E2B (visão)** → ~4.2 GB + ~1 GB overhead + KV cache (~5.2 GB total at 4K ctx)
- **Gemma 4 E4B (texto)** → ~4.9 GB + ~1 GB overhead + KV cache (~5.9 GB total at 4K ctx)
- **Gemma 4 E4B (visão)** → ~5.9 GB + ~1 GB overhead + KV cache (~6.9 GB total at 4K ctx)
- **Dolphin3.0 Llama3.2-3B Q4_K_M** → ~1.9 GB + ~1 GB overhead + KV cache (~2.9 GB total at 4K ctx)

Com 16 GB você pode rodar **1 modelo Bonsai 27B + 2 modelos pequenos** simultaneamente, mas **nunca os dois Bonsai ao mesmo tempo** (cada um precisa de ~5-8 GB). Os Gemma 4 E2B/E4B são leves o suficiente para rodar lado a lado com outros modelos. [skywork](https://skywork.ai/blog/models/qwen2-5-1-5b-instruct-gguf-free-chat-online-skywork-ai/)

To measure real performance:

```bash
# a test generation with statistics
./llama.cpp/build/ReleaseOV/bin/llama-cli \
  -m ./models/text/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  -p "Throughput test." -n 256 -t 8 -c 4096 -b 256 -ngl 0 \
  --log-disable
```

Observe total time / tokens generated.

***
## 📁 10. Final Project Organization
On WSL2, the consolidated structure is:

```text
~/llm-stack
  ├── llama.cpp/               # Compiled binaries (llama-server, llama-cli em build/ReleaseOV/bin/)
  ├── models/
  │   ├── text/                # Text .gguf (Qwen, Gemma2, Gemma 4 E2B/E4B, Dolphin3.0)
  │   ├── code/                # Code .gguf (Qwen Coder, Ministral, Ministral + mmproj)
  │   ├── vision/              # Vision .gguf (Qwen-VL + mmproj)
  │   └── bonsai/              # Bonsai 27B 1-bit + Ternary Bonsai 27B
  ├── vision/                  # venv + qwen_vl_server.py
  ├── gateway/                 # venv + gateway_server.py + models.yaml
  ├── scripts/                 
  │   ├── manage.sh            # Main control script (Start/Stop/Status)
  │   └── chat.sh              # Script for interactive CLI chat
  ├── external_config.yaml     # Configurations for n8n / external tools
  └── logs/                    # Log files (optional)
```

***

## 🔄 11. Automation and Daily Usage
It is not necessary to run manual `llama.cpp` or `tmux` commands. Use the scripts in the `scripts/` folder.

### 11.1. Service Manager (`manage.sh`)
The `manage.sh` is the "remote control" of your stack.

**Commands:**
- `./scripts/manage.sh start [target]`   → Start a model/service in the background.
- `./scripts/manage.sh stop [target]`    → Stop a specific service.
- `./scripts/manage.sh stop-all`         → Shut everything down and free RAM.
- `./scripts/manage.sh status`           → List which sessions are active.

### 11.2. Interactive Terminal Chat (`chat.sh`)
If you want to chat with a model directly from the terminal (without going through the network/gateway), use `chat.sh`.

**Usage:**
```bash
./scripts/chat.sh [model]
```
*Note: The chosen model must be STOPPED in `manage.sh` to avoid memory conflict.*

---

## ⚠️ 12. Common problems and how to solve them
### 12.1. `llama.cpp` compilation errors
- Missing `cmake` or `build-essential` → install with `apt` as above. [github](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- Old `cmake` version → if Ubuntu is too old, install a newer `cmake` via `snap` or official script (in your case 22.04 is fine).
### 12.2. “out of memory” when loading model
- Happens if you try to bring up too many models or with too large a context.  
- Adjust:
  - Reduce `-c 4096` to `2048` on the less critical model.  
  - Close some servers (e.g., keep only 2 active).  
  - Ensure there are no other heavy processes on Windows competing for RAM with WSL2.
### 12.3. Port already in use
- Message: `bind: Address already in use`.  
- Use `ss -tulpen | grep 8001` to see who is on the port; kill the process or change the port in the command.
### 12.4. Model not loading (file not found)
- Check the full path:
  - `ls ~/llm-stack/models/text`  
  - Compare with the path in `-m`.  
- If downloading again, check that `wget` did not save with a different name (`?download=1` etc.).
### 12.5. Gateway returning 400 “Unknown model”
- The `"model"` field in the JSON must match the key in `models.yaml` (`qwen2.5-1.5b-instruct`, `qwen2.5-coder-1.5b`, etc.).
### 12.6. Qwen‑VL slow
- VLMs are heavier than pure LLMs; use for occasional tasks (captioning, not long chat). [huggingface](https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF)
```