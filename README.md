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
  ├── llama.cpp/               # Source code and compiled binaries (build/CPU/)
  ├── models/                  # .gguf files organized by category
  │   ├── text/                # Gemma 4 (E2B/E4B), Dolphin3.0
  │   ├── code/                # Qwen2.5-Coder, LFM 2.5, Nanbeige4.2, MiniCPM5-1B
  │   ├── vision/              # Qwen2.5-VL (Model + mmproj)
  │   └── bonsai/              # Bonsai 27B 1-bit + Ternary Bonsai 27B
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
```

This covers compilation of `llama.cpp` (GGML nativo CPU), Python for the gateway, and `tmux` to run in the background. [github](http://github.com/ggml-org/llama.cpp)

If you want Node.js (optional, not used here):

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

***
## 🧠 2. Clone and compile `llama.cpp` (GGML nativo CPU)
We'll use the official repository `ggml-org/llama.cpp`. [github](http://github.com/ggml-org/llama.cpp)
### 2.1. Clone
```bash
cd ~/llm-stack
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```
### 2.2. Build (Release, GGML nativo CPU)
```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j"$(nproc)"
```

This generates binaries in `~/llm-stack/llama.cpp/build/` (`llama-cli`, `llama-server`, etc.). [github](http://github.com/ggml-org/llama.cpp)

> Note: we are not enabling CUDA/OpenVINO here to keep things simple and CPU-focused. The OpenVINO backend was removed from this stack: it produced corrupted output / shape errors on Qwen models, and no current model uses it — every model runs on the native GGML CPU build. If you ever want to try OpenVINO again, recompile with `-DGGML_OPENVINO=ON` into a separate build dir.

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

### 3.2. Code
#### Qwen2.5‑Coder‑3B‑Instruct‑Q4_K_M.gguf

GGUF repo oficial: `Qwen/Qwen2.5-Coder-3B-Instruct-GGUF`. [huggingface](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF)

Arquivo: `qwen2.5-coder-3b-instruct-q4_k_m.gguf` (2.0 GB, Q4_K_M, ~3B params, recomendado). [huggingface](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF)

```bash
cd ~/llm-stack/models/code
wget -O qwen2.5-coder-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf
```

#### Liquid LFM 2.5-1.2B-Instruct (Q8_0)

O **LFM 2.5 (Liquid Foundation Model 2.5)** é um modelo compacto de 1.2B parâmetros da LiquidAI, otimizado para código e raciocínio com qualidade excepcional graças à quantização Q8_0 (8-bit, quase sem perda).

Repo: `LiquidAI/LFM2.5-1.2B-Instruct-GGUF`. [huggingface](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF)

- **Arquivo:** `LFM2.5-1.2B-Instruct-Q8_0.gguf` (1.2 GB)
- **Qualidade:** Excelente (Q8_0 = quase perda zero, ~99% do FP16)
- **RAM (4K ctx):** ~2.2 GB ✅ cabe com folga em 8 GB WSL2
- **Velocidade CPU (i5):** ~25-40 tok/s ⚡ Muito rápido
- **Sem thinking mode** (modelo base, sem tokens especiais de raciocínio)

```bash
cd ~/llm-stack/models/code

wget -O LFM2.5-1.2B-Instruct-Q8_0.gguf \
  https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q8_0.gguf
```

#### Nanbeige4.2-3B (Q4_K_M)

O **Nanbeige4.2-3B** é um modelo da OWAO baseado no Nanbeige, com 3B parâmetros. Otimizado para código, raciocínio e tarefas gerais com qualidade excelente em Q4_K_M.

Repo: `owao/Nanbeige4.2-3B-GGUF`. [huggingface](https://huggingface.co/owao/Nanbeige4.2-3B-GGUF)

- **Arquivo:** `Nanbeige4.2-3B-Q4_K_M.gguf` (2.4 GB)
- **Qualidade:** Muito boa (Q4_K_M = balanço qualidade/RAM)
- **RAM (4K ctx):** ~3.4 GB ✅ cabe em 8 GB WSL2
- **Velocidade CPU (i5):** ~18-30 tok/s ⚡ Rápido
- **Com thinking mode** (tags `<think>`/`</think>` ativadas por padrão)
- Use `nanbeige-3b-nothink` para desligar o thinking e obter respostas mais diretas

```bash
cd ~/llm-stack/models/code

wget -O Nanbeige4.2-3B-Q4_K_M.gguf \
  https://huggingface.co/owao/Nanbeige4.2-3B-GGUF/resolve/main/Nanbeige4.2-3B-Q4_K_M.gguf
```

#### MiniCPM5-1B-Agentic-Tooluse (Q8_0)

O **MiniCPM5-1B-Agentic-Tooluse** é um modelo compacto de 1B parâmetros da OpenBMB, fine-tuned com Nemotron-DPO para ferramentas XML e fluxos agentivos. Baseado em LlamaForCausalLM, nativo 128K de contexto.

Repo: `ewinregirgojr/MiniCPM5-1B-Agentic-Tooluse-GGUF`. [huggingface](https://huggingface.co/ewinregirgojr/MiniCPM5-1B-Agentic-Tooluse-GGUF)

- **Arquivo:** `MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf` (1.1 GB)
- **Qualidade:** Excelente (Q8_0 = quase perda zero, ~99% do FP16)
- **RAM (4K ctx):** ~2.1 GB ✅ cabe com folga em 8 GB WSL2
- **Velocidade CPU (i5):** ~30-50 tok/s ⚡ Muito rápido
- **Com thinking mode** (tags `<think>`/`</think>` via `enable_thinking`)
- Use `minicpm5-1b-nothink` para desligar o thinking e obter respostas mais diretas
- **128K contexto nativo** (recomendado manter 4K-8K para CPU)

```bash
cd ~/llm-stack/models/code

wget -O MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf \
  https://huggingface.co/ewinregirgojr/MiniCPM5-1B-Agentic-Tooluse-GGUF/resolve/main/MiniCPM5-1B-Agentic-Tooluse-Nemotron-DPO.Q8_0.gguf
```

### 3.3. Vision

#### Qwen2.5-VL-3B-Uncensored (Q4_K_M)

O **Qwen2.5-VL-3B-Uncensored** (também chamado de Abliterated) é uma versão do Qwen2.5-VL-3B-Instruct com as recusas de conteúdo removidas (uncensored). Usa `llama.cpp` multimodal nativo com suporte a imagens via mmproj.

Repo: `mradermacher/Qwen2.5-VL-3B-Instruct-abliterated-GGUF`. [huggingface](https://huggingface.co/mradermacher/Qwen2.5-VL-3B-Instruct-abliterated-GGUF)

- **Modelo:** `qwen2.5-vl-3b-uncensored-Q4_K_M.gguf` (1.8 GB) — Q4_K_M, qualidade/recomendado
- **mmproj:** `qwen2.5-vl-3b-uncensored-mmproj-Q8_0.gguf` (1.3 GB Q8_0, visão)
- **RAM (4K ctx):** ~3.1 GB total (1.8 GB modelo + 1.3 GB mmproj) ✅ cabe em 8 GB WSL2
- **Velocidade CPU (i5):** ~15-25 tok/s
- **Short name (chat.sh/manage.sh):** `qwen-vl-3b-uncensored`
- **Porta:** 8010

> **Sobre mmproj Q8_0:** O repositório `mradermacher` não inclui arquivos mmproj. Usamos o mmproj Q8_0 (f16 convertido) do repo `lmstudio-community/Qwen2.5-VL-3B-Instruct-GGUF`, que é totalmente compatível por ser a mesma arquitetura. O mmproj (projeção do encoder de visão) só existe em formatos f16/Q8_0 — não há versão Q4_K_M.

```bash
cd ~/llm-stack/models/vision

# Modelo principal (Q4_K_M)
wget -O qwen2.5-vl-3b-uncensored-Q4_K_M.gguf \
  https://huggingface.co/mradermacher/Qwen2.5-VL-3B-Instruct-abliterated-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-abliterated.Q4_K_M.gguf

# mmproj (compatível, Q8_0)
wget -O qwen2.5-vl-3b-uncensored-mmproj-Q8_0.gguf \
  https://huggingface.co/lmstudio-community/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-model-f16.gguf
```

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

> **Dica de RAM:** Use as variantes **sem mmproj** (gemma-4-e2b, gemma-4-e4b) para economizar ~1 GB quando não for usar imagens. As variantes **-vision** (gemma-4-e2b-vision, gemma-4-e4b-vision) carregam o mmproj para suporte multimodal.

> **Sobre thinking mode:** Gemma 4 tem pensamento interno via `<|think|>` tokens. Use as variantes `-nothink` para desligar e obter respostas mais diretas.

***
## 🧠 2b. What is Q4_K_M / IQ4_XS?
- **Q4_K_M**: 4‑bit quantization in "super‑blocks" with per‑block statistics, 4.5 bits per weight, optimal quality vs RAM trade‑off – the normally recommended option for general use. [huggingface](https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF)
- **IQ4_XS**: "imatrix" super‑compact 4‑bit variant, with additional compression and small quality loss, often used in small models to reduce footprint.

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
- `qwen-coder-3b`       → Port 8003 (Qwen2.5-Coder 3B Q4_K_M ~12-20 tok/s)
- `qwen-vl-3b-uncensored` → Port 8010 (Qwen2.5-VL-3B abliterado Q4_K_M ~15-25 tok/s)
- `bonsai-27b`         → Port 8011 (Bonsai 27B 1-bit ~4-8 tok/s)
- `bonsai-27b-nothink` → Port 8013 (Bonsai 27B, thinking OFF)
- `ternary-bonsai-27b` → Port 8012 (Ternary Bonsai 27B ~2-5 tok/s)
- `ternary-bonsai-27b-nothink` → Port 8014 (Ternary Bonsai 27B, thinking OFF)
- `gemma-4-e2b`        → Port 8021 (Gemma 4 E2B 2.3B ~22-35 tok/s, texto puro)
- `gemma-4-e2b-nothink`→ Port 8023 (Gemma 4 E2B, thinking OFF)
- `gemma-4-e2b-vision` → Port 8031 (Gemma 4 E2B + visão)
- `gemma-4-e2b-vision-nothink`→ Port 8033 (Gemma 4 E2B + visão, thinking OFF)
- `gemma-4-e4b`        → Port 8022 (Gemma 4 E4B 4.5B ~10-15 tok/s, texto puro)
- `gemma-4-e4b-nothink`→ Port 8024 (Gemma 4 E4B, thinking OFF)
- `gemma-4-e4b-vision` → Port 8032 (Gemma 4 E4B + visão)
- `gemma-4-e4b-vision-nothink`→ Port 8034 (Gemma 4 E4B + visão, thinking OFF)
- `dolphin3-3b`        → Port 8041 (Dolphin3.0 Llama3.2-3B Q4_K_M ~20-30 tok/s)
- `lfm2.5-1.2b`        → Port 8061 (Liquid LFM 2.5-1.2B Q8_0 ~25-40 tok/s)
- `nanbeige-3b`        → Port 8071 (Nanbeige4.2-3B Q4_K_M ~18-30 tok/s)
- `nanbeige-3b-nothink`→ Port 8073 (Nanbeige4.2-3B, thinking OFF)
- `minicpm5-1b`        → Port 8081 (MiniCPM5-1B Nemotron-DPO Q8_0 ~30-50 tok/s)
- `minicpm5-1b-nothink`→ Port 8083 (MiniCPM5-1B, thinking OFF)
- `gateway`            → Port 9000 (The Central Router)

### 4.2. Useful Manager commands
- `./scripts/manage.sh status`      → See what is running.
- `./scripts/manage.sh stop-all`    → Kill all services and free RAM.
- `./scripts/manage.sh stop nanbeige-3b` → Stop only a specific model.

***
## 🌐 5. OpenAI‑like HTTP API of `llama-server`
The `llama-server` exposes OpenAI‑compatible endpoints, such as `/v1/chat/completions`. [github](http://github.com/ggml-org/llama.cpp)

Quick test (on WSL2):

```bash
curl http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-coder-3b",
    "messages": [
      {"role": "user", "content": "Write a C function that reverses a string in-place."}
    ],
    "max_tokens": 128
  }'
```

This should already return JSON in OpenAI format. [learn.arm](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama-cpu/llama-server/)

> The `"model"` field here is ignored by `llama-server` (it is already "fixed" in the binary), but we will use this field in the *gateway* for routing.

***
## 🧠 10. Vision – servers
### 10.1. Vision via llama.cpp (GGUF multimodal nativo)
O `llama.cpp` agora suporta modelos multimodais nativamente via GGUF. O **Qwen2.5-VL-3B-Uncensored** roda diretamente no `llama-server` com suporte a imagens através do mmproj.

**Comando (via manage.sh):**
```bash
./scripts/manage.sh start qwen-vl-3b-uncensored
```

**Chat interativo:**
```bash
./scripts/chat.sh qwen-vl-3b-uncensored
# Depois use /image caminho/da/imagem.jpg dentro do chat
```

**Teste via HTTP:**
```bash
curl http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-3b-uncensored",
    "messages": [
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file:///path/to/image.jpg"}},
        {"type": "text", "text": "Describe this image in detail."}
      ]}
    ],
    "max_tokens": 256
  }'
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
  qwen-coder-3b:
    type: code
    endpoint: http://localhost:8003

  qwen-vl-3b-uncensored:
    type: vision
    endpoint: http://localhost:8010

  bonsai-27b:
    type: text
    endpoint: http://localhost:8011

  bonsai-27b-nothink:
    type: text
    endpoint: http://localhost:8013

  ternary-bonsai-27b:
    type: text
    endpoint: http://localhost:8012

  ternary-bonsai-27b-nothink:
    type: text
    endpoint: http://localhost:8014

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

  dolphin3-3b:
    type: text
    endpoint: http://localhost:8041

  lfm2.5-1.2b:
    type: code
    endpoint: http://localhost:8061

  nanbeige-3b:
    type: code
    endpoint: http://localhost:8071

  nanbeige-3b-nothink:
    type: code
    endpoint: http://localhost:8073

  minicpm5-1b:
    type: code
    endpoint: http://localhost:8081

  minicpm5-1b-nothink:
    type: code
    endpoint: http://localhost:8083
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
### 8.1. Chat with bonsai-27b (via gateway)
```bash
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bonsai-27b",
    "messages": [
      {"role": "system", "content": "You are a concise technical assistant."},
      {"role": "user", "content": "Briefly explain what a syscall is."}
    ],
    "max_tokens": 128,
    "temperature": 0.4
  }' | jq
```
### 8.2. Chat with Nanbeige4.2-3B
```bash
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nanbeige-3b",
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
    "model": "qwen-coder-3b",
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

- **Qwen2.5‑Coder‑3B‑Q4_K_M** → ~2.0 GB + ~1 GB overhead + KV cache (~3.0 GB total at 4K ctx)
- **Bonsai 27B 1-bit** → ~3.9 GB + ~1.3 GB overhead + KV cache (~5.2 GB total at 4K ctx)
- **Ternary Bonsai 27B** → ~7.2 GB + ~1.2 GB overhead + KV cache (~8.4 GB total at 4K ctx)
- **Gemma 4 E2B (texto)** → ~3.2 GB + ~1 GB overhead + KV cache (~4.2 GB total at 4K ctx)
- **Gemma 4 E2B (visão)** → ~4.2 GB + ~1 GB overhead + KV cache (~5.2 GB total at 4K ctx)
- **Gemma 4 E4B (texto)** → ~4.9 GB + ~1 GB overhead + KV cache (~5.9 GB total at 4K ctx)
- **Gemma 4 E4B (visão)** → ~5.9 GB + ~1 GB overhead + KV cache (~6.9 GB total at 4K ctx)
- **Dolphin3.0 Llama3.2-3B Q4_K_M** → ~1.9 GB + ~1 GB overhead + KV cache (~2.9 GB total at 4K ctx)
- **LFM 2.5 1.2B Q8_0** → ~1.2 GB + ~1 GB overhead + KV cache (~2.2 GB total at 4K ctx)
- **Nanbeige4.2-3B Q4_K_M** → ~2.4 GB + ~1 GB overhead + KV cache (~3.4 GB total at 4K ctx)
- **MiniCPM5-1B Q8_0** → ~1.1 GB + ~1 GB overhead + KV cache (~2.1 GB total at 4K ctx)
- **Qwen2.5-VL-Uncensored Q4_K_M + mmproj** → ~1.8 GB + 1.3 GB + KV cache (~3.1 GB total at 4K ctx)

Com 16 GB você pode rodar **1 modelo Bonsai 27B + 2 modelos pequenos** simultaneamente, mas **nunca os dois Bonsai ao mesmo tempo** (cada um precisa de ~5-8 GB). Os Gemma 4 E2B/E4B são leves o suficiente para rodar lado a lado com outros modelos. [skywork](https://skywork.ai/blog/models/qwen2-5-1-5b-instruct-gguf-free-chat-online-skywork-ai/)

To measure real performance:

```bash
# a test generation with statistics
./llama.cpp/build/CPU/bin/llama-cli \
  -m ./models/code/qwen2.5-coder-3b-instruct-q4_k_m.gguf \
  -p "Throughput test." -n 256 -t 8 -c 4096 -b 256 -ngl 0 \
  --log-disable
```

Observe total time / tokens generated.

***
## 📁 10. Final Project Organization
On WSL2, the consolidated structure is:

```text
~/llm-stack
  ├── llama.cpp/               # Compiled binaries (llama-server, llama-cli em build/CPU/bin/)
  ├── models/
  │   ├── text/                # Text .gguf (Gemma 4 E2B/E4B, Dolphin3.0)
  │   ├── code/                # Code .gguf (Qwen Coder, LFM 2.5, Nanbeige4.2, MiniCPM5)
  │   ├── vision/              # Vision .gguf (Qwen-VL + mmproj)
  │   └── bonsai/              # Bonsai 27B 1-bit + Ternary Bonsai 27B
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
- The `"model"` field in the JSON must match the key in `models.yaml` (`qwen-coder-3b`, `nanbeige-3b`, `bonsai-27b`, etc.).
### 12.6. Qwen‑VL slow
- VLMs are heavier than pure LLMs; use for occasional tasks (captioning, not long chat). [huggingface](https://huggingface.co/prithivMLmods/Qwen2.5-VL-Abliterated-Caption-GGUF)
```